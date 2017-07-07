"""
=====================================
The :mod:`mpi_array.logging` Module
=====================================

Default initialisation of python logging.


Some simple wrappers of python built-in :mod:`logging` module
for :mod:`mpi_array` logging.

.. currentmodule:: mpi_array.logging

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   SplitStreamHandler - A :obj:`logging.StreamHandler` which splits errors and warnings to *stderr*.
   initialise_loggers - Initialises handlers and formatters for loggers.
   get_rank_logger - Returns :obj:`logging.Logger` for MPI ranks.
   get_root_logger - Returns :obj:`logging.Logger` for MPI root-rank.
   LoggerFactory - Factory class for generating :obj:`logging.Logger` objects.

Attributes
==========

.. autodata:: logger_factory

"""

from __future__ import absolute_import

import mpi4py.MPI as _mpi
import sys
import logging as _builtin_logging
from logging import *  # noqa: F401,F403
import copy as _copy

if (sys.version_info[0] <= 2):
    from sets import Set as set


class _Python2SplitStreamHandler(_builtin_logging.Handler):

    """
    A python :obj:`logging.handlers` :samp:`Handler` class for
    splitting logging messages to different streams depending on
    the logging-level.
    """

    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=_builtin_logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.

        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages.
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        _builtin_logging.Handler.__init__(self)

    def emit(self, record):
        """
        Mostly copy-paste from :obj:`logging.StreamHandler`.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            fs = "%s\n"

            try:
                if (isinstance(msg, unicode) and  # noqa: F405
                        getattr(stream, 'encoding', None)):
                    ufs = fs.decode(stream.encoding)
                    try:
                        stream.write(ufs % msg)
                    except UnicodeEncodeError:
                        stream.write((ufs % msg).encode(stream.encoding))
                else:
                    stream.write(fs % msg)
            except UnicodeError:
                stream.write(fs % msg.encode("UTF-8"))

            stream.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class _Python3SplitStreamHandler(_builtin_logging.Handler):

    """
    A python :obj:`logging.handlers` :samp:`Handler` class for
    splitting logging messages to different streams depending on
    the logging-level.
    """

    terminator = '\n'

    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=_builtin_logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.

        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages.
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        _builtin_logging.Handler.__init__(self)

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.outStream and hasattr(self.outStream, "flush"):
                self.outStream.flush()
            if self.errStream and hasattr(self.errStream, "flush"):
                self.errStream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except:
            self.handleError(record)


if (sys.version_info[0] <= 2):
    class SplitStreamHandler(_Python2SplitStreamHandler):
        __doc__ = _Python2SplitStreamHandler.__doc__
        pass
else:
    class SplitStreamHandler(_Python3SplitStreamHandler):
        __doc__ = _Python3SplitStreamHandler.__doc__
        pass


def get_handlers(logger):
    """
    Returns the handler objects for the specified :obj:`logging.Logger`
    object. Searches up the parent tree to find handlers.

    :type logger: :obj:`logging.Logger`
    :param logger: Searches this :obj:`logging.Logger` and parents
       to find :obj:`logging.Handler` objects.

    :rtype: :obj:`list`
    :return: :obj:`list` of :obj:`logging.Handler` instance objects.
    """
    handler_list = logger.handlers
    if logger.propagate and (not isinstance(logger, _builtin_logging.RootLogger)):
        handler_list += \
            get_handlers(_builtin_logging.getLogger(".".join(logger.name.split(".")[:-1])))
    return handler_list


def get_handler_classes(logger):
    """
    Returns the handler classes of handler objects associated with the
    specified :obj:`logging.Logger` object. Searches up the parent tree
    to find handlers.

    :type logger: :obj:`logging.Logger`
    :param logger: Searches this :obj:`logging.Logger` and parents
       to find :obj:`logging.Handler` objects.

    :rtype: :obj:`list`
    :return: :obj:`list` of :obj:`logging.Handler` class objects.
    """
    return list(set([h.__class__ for h in get_handlers(logger)]))


class MultiLineFormatter(_builtin_logging.Formatter):

    """
    Over-rides :obj:`logging.Formatter.format` to format all lines
    of a mulit-line log message.
    """

    #: Defines multiple lines.
    multi_line_split_string = "\n"

    def format(self, record):
        """
        Converts record to formatted string, each line of a multi-line
        message string is individually formatted.
        """
        # take care of the substitutions in record.args first
        fs = _builtin_logging.Formatter.format(self, record)
        messages = fs.split(self.multi_line_split_string)
        s_list = [messages[0], ]
        # Now format each line individually (no substitutions).
        for msg in messages[1:]:
            single_line_record = _copy.copy(record)
            single_line_record.args = None
            single_line_record.msg = msg
            fs = _builtin_logging.Formatter.format(self, single_line_record)
            s_list.append(fs)

        return self.multi_line_split_string.join(s_list)


class LoggerFactory (object):

    """
    Factory for generating :obj:`logging.Logger` instances.
    """

    def __init__(self):
        """
        """
        pass

    def get_formatter(self, prefix_string="MPIARY|"):
        """
        Returns :obj:`logging.Formatter` object which produces messages
        with *time* and :samp:`prefix_string` prefix.

        :type prefix_string: :obj:`str` or :samp:`None`
        :param prefix_string: Prefix for all logging messages.
        :rtype: :obj:`logging.Formatter`
        :return: Regular formatter for logging.
        """
        if (prefix_string is None):
            prefix_string = ""
        formatter = \
            MultiLineFormatter(
                "%(asctime)s|" + prefix_string + "%(message)s",
                "%H:%M:%S"
            )

        return formatter

    def get_rank_logger(self, name, comm=None, ranks=None, rank_string="rank"):
        """
        Returns a :obj:`logging.Logger` object with time-stamp, :samp:`{comm}.Get_name()`
        and :samp:`{comm}.Get_rank()` in the message.

        :type name: :obj:`str`
        :param name: Name of logger (note that the name of logger actuallycreated will
           be :samp:`{name} + "." + {comm}.Get_name() + ".rank." + ("%04d" % {comm}.Get_rank())`).
        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: MPI communicator. Used for determining the rank of this process.
            If :samp:`None` uses :samp:`mpi4py.MPI.COMM_WORLD`.
        :type ranks: :obj:`None` or :obj:`list`-of-:obj:`int`
        :param ranks: Limits logging output to ranks specified in this list.
            If :samp:`None`, all ranks produce logging output.
        :rtype: :obj:`logging.Logger`
        :return: Logger object.
        """
        if comm is None:
            comm = _mpi.COMM_WORLD

        rank_logger = \
            _builtin_logging.getLogger(
                name +
                "." +
                comm.Get_name() +
                "." +
                rank_string +
                "." +
                (("%%0%dd" % (len(str(_mpi.COMM_WORLD.size - 1)),)) % comm.Get_rank())
            )
        # First search for handler classes.
        tmp_logger = _builtin_logging.getLogger(name)
        hander_classes = get_handler_classes(tmp_logger)
        # Don't propagate, new handler object instances are assigned at the leaf.
        rank_logger.propagate = False
        rank_logger.handlers = []
        f = \
            self.get_formatter(
                prefix_string=(
                    ("%%s|%%s%%0%dd|" % (len(str(_mpi.COMM_WORLD.size - 1)),))
                    %
                    (comm.Get_name(), rank_string, comm.Get_rank())
                )
            )
        if (ranks is None) or (comm.Get_rank() in ranks):
            for handler_class in hander_classes:
                h = handler_class()
                h.setFormatter(f)
                rank_logger.addHandler(h)
        else:
            # NullHandler is always silent
            rank_logger.addHandler(_builtin_logging.NullHandler())

        return rank_logger

    def get_root_logger(self, name, comm=None, root_rank=0):
        """
        Returns a :obj:`logging.Logger` object with time-stamp, :samp:`{comm}.Get_name()`
        and :samp:`{comm}.Get_rank()` in the message. Logging output limited to
        the MPI rank specified by :samp:`{root_rank}`.

        :type name: :obj:`str`
        :param name: Name of logger (note that the name of logger actually
           created will be :samp:`{name} + ".rank." + ("%04d" % {comm}.Get_rank())`).
        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: MPI communicator. Used for determining the rank of this process.
            If :samp:`None` uses :samp:`mpi4py.MPI.COMM_WORLD`.
        :type root_rank: :obj:`int`
        :param root_rank: Logging output is limited to this rank,
            the returned :obj:`logging.Logger` objects on
            other ranks have a :obj:`logging.NullHandler`.
        :rtype: :obj:`logging.Logger`
        :return: Logger object.
        """
        return self.get_rank_logger(name, comm, ranks=[root_rank, ], rank_string="root")


#: Factory for creating :obj:`logging.Logger` objects.
#: Can set value to different instance in order to
#: customise logging output.
logger_factory = LoggerFactory()


def get_rank_logger(name, comm=None, ranks=None):
    """
    Returns :obj:`logging.Logger` object for message logging.

    :type name: :obj:`str`
    :param name: Name of logger (note that the name of logger actually
       created will be :samp:`{name} + ".rank." + ("%04d" % {comm}.Get_rank())`).
    :type comm: :obj:`mpi4py.MPI.Comm`
    :param comm: MPI communicator. Used for determining the rank of this process.
        If :samp:`None` uses :samp:`mpi4py.MPI.COMM_WORLD`.
    :type ranks: :obj:`None` or :obj:`list`-of-:obj:`int`
    :param ranks: Limits logging output to ranks specified in this list.
        If :samp:`None`, all ranks produce logging output.
    :rtype: :obj:`logging.Logger`
    :return: Logger object.
    """
    return logger_factory.get_rank_logger(name=name, comm=comm, ranks=ranks)


def get_root_logger(name, comm=None, root_rank=0):
    """
    Returns a :obj:`logging.Logger` object with time-stamp, :samp:`{comm}.Get_name()`
    and :samp:`{comm}.Get_rank()` in the message. Logging output limited to
    the MPI rank specified by :samp:`{root_rank}`.

    :type name: :obj:`str`
    :param name: Name of logger (note that the name of logger actually
       created will be :samp:`{name} + ".rank." + ("%04d" % {comm}.Get_rank())`).
    :type comm: :obj:`mpi4py.MPI.Comm`
    :param comm: MPI communicator. Used for determining the rank of this process.
        If :samp:`None` uses :samp:`mpi4py.MPI.COMM_WORLD`.
    :type root_rank: :obj:`int`
    :param root_rank: Logging output is limited to this rank,
        the returned :obj:`logging.Logger` objects on
        other ranks have a :obj:`logging.NullHandler`.
    :rtype: :obj:`logging.Logger`
    :return: Logger object.
    """
    return logger_factory.get_root_logger(name=name, comm=comm, root_rank=root_rank)


def initialise_loggers(names, log_level=_builtin_logging.WARNING, handler_class=SplitStreamHandler):
    """
    Initialises specified loggers to generate output at the
    specified logging level. If the specified named loggers do not exist,
    they are created.

    :type names: :obj:`list` of :obj:`str`
    :param names: List of logger names.
    :type log_level: :obj:`int`
    :param log_level: Log level for messages, typically
       one of :obj:`logging.DEBUG`, :obj:`logging.INFO`, :obj:`logging.WARN`, :obj:`logging.ERROR`
       or :obj:`logging.CRITICAL`.
       See :ref:`levels`.
    :type handler_class: One of the :obj:`logging.handlers` classes.
    :param handler_class: The handler class for output of log messages,
       for example :obj:`SplitStreamHandler` or :obj:`logging.StreamHandler`.

    """
    for name in names:
        logr = _builtin_logging.getLogger(name)
        handler = handler_class()
        logr.addHandler(handler)
        logr.setLevel(log_level)


__all__ = [s for s in dir() if not s.startswith('_')]
