"""
=========================================
The :mod:`mpi_array.globale_ufunc` Module
=========================================

Defines :obj:`numpy.ufunc` functions for :obj:`mpi_array.globale.gndarray`.

Classes
=======

.. autosummary::
   :toctree: generated/

   GndarrayArrayUfuncExecutor - Creates :obj:`gndarray` outputs and forwards to `numpy.ufunc`.

Functions
=========

.. autosummary::
   :toctree: generated/

   get_dtype_and_ndim - Return :obj:`numpy.dtype` and :samp:`ndim` properties for an object.
   ufunc_result_type - Like :func:`numpy.result_type`.
   broadcast_shape - Calculates broadcast shape from sequence of shape arguments.
   shape_extend_dims - Prepend ones to 1D *shape* sequence to make it a specified dimension.
   gndarray_array_ufunc - A :obj:`numpy.ndarray` like distributed array.


"""

from __future__ import absolute_import

import sys as _sys
import numpy as _np
import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import logging as _logging  # noqa: E402,F401
from . import globale_creation as _globale_creation
from . import comms as _comms
from .distribution import ScalarLocaleExtent, ScalarGlobaleExtent, LocaleExtent, GlobaleExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def get_dtype_and_ndim(array_like):
    """
    Returns :samp:`(dtype, ndim)` pair for the given :samp:`{array_like}` argument.
    If the :samp:`{array_like}` has *both* :samp:`"dtype"` and :samp:`"ndim"`
    attributes, then the return tuple is :samp:`({array_like}.dtype, {array_like}.ndim)`.
    Otherwise,
    returns :samp:`(numpy.asanyarray({array_like}).dtype, numpy.asanyarray({array_like}).ndim)`.

    :type array_like: castable to :obj:`numpy.ndarray`
    :param array_like: Returns dtype and ndim for this object.
    :rtype: two element :obj:`tuple`
    :return: The :obj:`numpy.dtype` and integer :samp:`ndim` properties for :samp:`{array_like}`.

    Example::

       >>> get_dtype_and_ndim(1.0)
       (dtype('float64'), 0)
       >>> get_dtype_and_ndim((1.0, 2.0, 3.0, 4.0))
       (dtype('float64'), 1)
       >>> get_dtype_and_ndim([(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)])
       (dtype('float64'), 2)
    """
    dt, nd = None, None
    if not ((hasattr(array_like, "dtype") and hasattr(array_like, "ndim"))):
        array_like = _np.asanyarray(array_like)

    dt, nd = array_like.dtype, array_like.ndim

    return dt, nd


def ufunc_result_type(
    ufunc_types,
    inputs,
    outputs=None,
    casting="safe",
    input_match_casting="safe"
):
    """
    Attempts to calculate the result type from given ufunc :samp:`{inputs}`
    and ufunc types (:attr:`numpy.ufunc.types`).
    Like :obj:`numpy.result_type`, but
    handles :obj:`mpi_array.globale.gndarray` in the :samp:`{inputs}`
    and handles multiple :samp:`{outputs}` cases.

    :type ufunc_types: sequence of `str`
    :param ufunc_types: The :attr:`numpy.ufunc.types` attribute,
       e.g. :samp:`['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', ..., 'mm->m', 'mM->M', 'OO->O']`.
    :type inputs: sequence of :obj:`object`
    :param inputs: The inputs (e.g. :obj:`numpy.ndarray`, scalars
       or :obj:`mpi_array.globale.gndarray`) to a :obj:`numpy.ufunc` call.
    :type outputs: :samp:`None` or sequence of :obj:`object`
    :param outputs: The output arrays these are explicitly checked casting correctness.
    :type casting: :obj:`str` :samp:`{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}`
    :param casting: Casting mode applied to outputs. See :func:`numpy.can_cast`.
    :type input_match_casting: :obj:`str` :samp:`{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}`
    :param input_match_casting: Casting mode applied to match :samp:`{ufunc_types}` inputs
       with the :samp:`{inputs}`. See :func:`numpy.can_cast`.
    :rtype: :obj:`tuple` of :obj:`numpy.dtype`
    :return: A tuple of :obj:`numpy.dtype` indicating the output types produced for
       the given inputs.
    :raises ValueError: If the the inputs (and outputs) cannot be cast to an
       appropriate element of :samp:`{ufunc_types}`.

    Example::

       >>> import numpy as np
       >>> import mpi_array as mpia
       >>> inp = (
       ... np.zeros((10,10,10), dtype='float16'),
       ... 16.0,
       ... mpia.zeros((10,10,10), dtype='float32'),
       ... )
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp)
       (dtype('float32'), dtype('bool'))
       >>> out = (mpia.zeros((10,10,10), dtype="float64"),)
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
       (dtype('float64'), dtype('bool'))
       >>> out += (mpia.zeros((10, 10, 10), dtype="uint16"),)
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
       (dtype('float64'), dtype('uint16'))
       >>> mpia.free_all(inp + out)
    """
    logger = _logging.get_rank_logger(__name__)
    result_dtypes = None
    ufunc_in_types = tuple(in2out_str.split("->")[0] for in2out_str in ufunc_types)
    ufunc_in_dtypes = \
        _np.asarray(
            tuple(
                tuple(_np.dtype(c) for c in ufunc_in_types[i])
                for i in range(len(ufunc_in_types))
            )
        )
    ufunc_out_types = tuple(in2out_str.split("->")[1] for in2out_str in ufunc_types)
    ufunc_out_dtypes = \
        _np.asarray(
            tuple(
                tuple(_np.dtype(c) for c in ufunc_out_types[i])
                for i in range(len(ufunc_out_types))
            )
        )

    in_dtypes_and_ndims = \
        _np.asarray(tuple(get_dtype_and_ndim(input) for input in inputs))

    in_dtypes = in_dtypes_and_ndims[:, 0]
    in_ndims = in_dtypes_and_ndims[:, 1]

    logger.debug("inputs=%s", inputs)
    logger.debug("in_dtypes=%s", in_dtypes)
    logger.debug("in_ndims=%s", in_ndims)
    logger.debug("ufunc_in_dtypes=%s", ufunc_in_dtypes)

    out_dtypes = None
    if (outputs is not None) and (len(outputs) > 0):
        out_dtypes = \
            _np.asarray(
                tuple(
                    output.dtype
                    if hasattr(output, "dtype") else _np.asarray(output).dtype
                    for output in outputs
                )
            )

    idx = None
    idxs = _np.where(_np.logical_and.reduce(ufunc_in_dtypes == in_dtypes, axis=1))
    if len(idxs) > 0 and len(idxs[0]) > 0:
        idx = idxs[0][0]

    if idx is None:
        in_scalars_and_dtypes = \
            tuple(
                inputs[i]
                if in_ndims[i] <= 0 else in_dtypes[i]
                for i in range(len(inputs))
            )
        idxs = \
            _np.where(
                _np.asarray(
                    tuple(
                        _np.all(
                            tuple(
                                _np.can_cast(
                                    in_scalars_and_dtypes[j],
                                    ufunc_in_dtypes[i, j],
                                    casting=input_match_casting
                                )
                                for j in range(ufunc_in_dtypes.shape[1])
                            )
                        )
                        for i in range(ufunc_in_dtypes.shape[0])
                    )
                )
            )
        if len(idxs) > 0 and len(idxs[0]) > 0:
            idx = idxs[0][0]

    if idx is not None:
        ufunc_out_dtypes_for_in = ufunc_out_dtypes[idx]
        if (
            (out_dtypes is not None)
            and
            _np.any(ufunc_out_dtypes_for_in[:len(out_dtypes)] != out_dtypes)
        ):
            if (
                _np.any(
                    tuple(
                        not _np.can_cast(ufunc_out_dtypes_for_in[i], out_dtypes[i], casting=casting)
                        for i in range(len(out_dtypes))
                    )
                )
            ):
                raise ValueError(
                    "Could not cast ufunc-output-types %s to desired output-types = %s."
                    %
                    (
                        tuple(ufunc_out_dtypes_for_in),
                        tuple(out_dtypes)
                    )
                )
        if out_dtypes is None:
            out_dtypes = _np.array((), dtype='O')
        result_dtypes = \
            tuple(
                out_dtypes.tolist()
                +
                ufunc_out_dtypes_for_in[len(out_dtypes):].tolist()
            )
    else:
        raise ValueError(
            "Could not cast (with input_match_casting='%s') inputs types = %s to ufunc types=\n%s"
            %
            (input_match_casting, in_dtypes, ufunc_in_dtypes, )
        )

    return result_dtypes


def broadcast_shape(*shape_args):
    """
    Returns
    the :mod:`numpy` `broadcast <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    shape for the give shape arguments.

    :type shape1, shape2, ...: sequence of `int`
    :param shape1, shape2, ...: Array shapes to be broadcast.
    :rtype: sequence of `int`
    :return: The broadcast shape.

    Examples::

        >>> broadcast_shape((4,), (4,))
        (4,)
        >>> broadcast_shape((4, 1), (1, 5))
        (4, 5)
        >>> broadcast_shape((4, 1, 3, 7), (1, 8, 1, 7))
        (4, 8, 3, 7)
        >>> broadcast_shape((3, 7), ())
        (3, 7)
    """
    ndim = _np.max(tuple(len(shape) for shape in shape_args))

    bcast_shape = ()
    if ndim > 0:
        ndim_shapes = \
            _np.asarray(tuple((1,) * (ndim - len(shape)) + tuple(shape) for shape in shape_args))
        bcast_shape = _np.amax(ndim_shapes, axis=0)

        if (_np.any(_np.logical_and(ndim_shapes != 1, ndim_shapes != bcast_shape))):
            raise ValueError(
                "shape mismatch - objects cannot be broadcast to a single shape:\n%s"
                %
                (shape_args,)
            )

        bcast_shape = tuple(bcast_shape)

    return bcast_shape


def shape_extend_dims(ndim, shape):
    """
    Returns :obj:`shape` pre-prepended with ones so returned 1D array has length :samp:`{ndim}`.

    :type ndim: :obj:`int`
    :param ndim: Length of returned 1D sequence.
    :type shape: sequence of :obj:`object`
    :param shape: Length of returned 1D sequence.
    :rtype: :obj:`tuple`
    :return: Sequence pre-pended with one elements so that sequence length equals :samp:`{ndim}`.

    Example::

       >>> shape_extend_dims(5, (3, 1, 5))
       (1, 1, 3, 1, 5)
       >>> shape_extend_dims(3, (3, 1, 5))
       (3, 1, 5)
       >>> shape_extend_dims(1, (3, 1, 5))
       (3, 1, 5)

    """
    return (1,) * (ndim - len(shape)) + tuple(shape)


def get_extents(input, locale_info):
    """
    Returns a :samp:`(locale_extent, globale_extent)` pair for
    the given :samp:`input`, where :samp:`locale_extent` is
    a :obj:`mpi_array.distribution.LocaleExtent` instance and :samp:`globale_extent` is
    a :obj:`mpi_array.distribution.GlobaleExtent` instance.

    :type input: scalar, array like or :obj:`mpi_array.globale.gndarray`
    :param input: Return extents for this input.
    :type locale_info: :obj:`mpi_array.comms.ThisLocaleInfo`
    :param locale_info: The rank info required for constructing
        a :obj:`mpi_array.distribution.LocaleExtent` instance
        for :samp:`input` types which are not :obj:`mpi_array.globale.gndarray`.
    :rtype: :obj:`tuple`
    :return: A :samp:`(locale_extent, globale_extent)` pair indicating the
       extents of the :samp:`{input}` array-like.
    """
    locale_extent = None
    globale_extent = None
    if not (hasattr(input, "shape") and hasattr(input, "ndim")):
        input = _np.asanyarray(input)
    if hasattr(input, "lndarray_proxy") and hasattr(input, "distribution"):
        locale_extent = input.lndarray_proxy.locale_extent
        globale_extent = input.distribution.globale_extent
    elif input.ndim > 0:
        start = (0,) * input.ndim
        globale_extent = GlobaleExtent(start=start, stop=input.shape)
        locale_extent = \
            LocaleExtent(
                peer_rank=locale_info.peer_rank,
                inter_locale_rank=locale_info.inter_locale_rank,
                globale_extent=globale_extent,
                start=start,
                stop=input.shape
            )
    else:
        locale_extent = \
            ScalarLocaleExtent(
                peer_rank=locale_info.peer_rank,
                inter_locale_rank=locale_info.inter_locale_rank
            )
        globale_extent = ScalarGlobaleExtent()

    return (locale_extent, globale_extent)


def calc_matching_locale_slices(out_locale_extent, out_globale_extent, inp_locale_extents):
    """
    Returns :obj:`tuple` of :obj:`slice` (one tuple for each pair-element
    in :samp:`{inp_locale_extents}`). The returned *slices* indicate the
    portion of the corresponding input extent which broadcasts
    to the output extent :samp:`{out_locale_extent}`.

    Assumes :samp:`{out_locale_extent}.ndim >= {inp_locale_extents}[i].ndim`
    for :samp:`i in range(0, len({inp_locale_extents})`.

    :type out_locale_extent: :obj:`mpi_array.distribution.LocaleExtent`
    :param out_locale_extent: A locale extent of the output array.
    :type out_globale_extent: :obj:`mpi_array.distribution.GlobaleExtent`
    :param out_globale_extent: The globale extent of the output :obj:`mpi_array.globale.gndarray`.
    :type inp_locale_extents: sequence of extent pairs
    :param inp_locale_extents: This is the sequence
       of :samp:`(inp_locale_extent, inp_globale_extent)` pairs, one pair for
       each ufunc input.
    :rtype: :obj:`tuple` of :obj:`tuple` elements
    :return: For each pair :samp:`(inp_locale_extent, inp_globale_extent)`
       in :samp:`{inp_locale_extents}` returns a :obj:`tuple`-of-:obj:`slice`
       indicating the portion of :samp:`inp_locale_extent` which is to be broadcast
       with :samp:`{out_locale_extent}`. Tuple indices are globale.
    """
    slice_list = []
    out_loc_start = out_locale_extent.start
    out_loc_shape = out_locale_extent.shape
    for inp_loc, inp_glb in inp_locale_extents:
        slc_tuple = None
        if inp_glb.ndim >= 1:
            inp_glb_shape = inp_glb.shape
            inp_loc_start = inp_loc.start
            inp_loc_shape = inp_loc.shape
            inp_slc_start = _np.zeros_like(inp_loc_start)
            inp_slc_shape = inp_loc_shape.copy()

            slc_tuple = []
            for a in range(-1, -(len(inp_loc_shape) + 1), -1):
                if inp_glb_shape[a] == 1:
                    inp_slc_start[a] = 0
                    inp_slc_shape[a] = 1
                else:
                    inp_slc_start[a] = out_loc_start[a]
                    inp_slc_shape[a] = out_loc_shape[a]
                slc = slice(inp_slc_start[a], inp_slc_start[a] + inp_slc_shape[a])
                slc_tuple.insert(0, slc)
            slc_tuple = tuple(slc_tuple)
        slice_list.append(slc_tuple)

    return tuple(slice_list)


def calc_matching_peer_rank_slices(out_slice, inp_arys):
    """
    For each input array in :samp:`{inp_arys}, calculates the portion
    which broadcasts to the :samp:`{out_slice}`.
    Returns :obj:`tuple` of :obj:`slice` (one tuple for each array/scalar element
    in :samp:`{inp_arys}`). The returned *slices* indicate the
    portion of the input which matches the specified :samp:`{out_slice}`
    for broadcasting.

    Assumes :samp:`len({out_slice}) >= {inp_arys}[i].ndim`
    for :samp:`i in range(0, len({inp_arys})`.

    :type out_slice: :obj:`tuple` of :obj:`slice`
    :param out_slice: Slice indicating a portion (sub-array) of an output array.
    :type inp_arys: Sequence of :obj:`numpy.ndarray`
    :param inp_arys: The ufunc input arrays.
    """
    slice_list = []
    for inp_ary in inp_arys:
        slc_tuple = None
        if hasattr(inp_ary, "ndim") and (inp_ary.ndim >= 1):
            inp_shape = _np.array(inp_ary.shape)
            inp_slc_start = _np.zeros_like(inp_shape)
            inp_slc_stop = inp_slc_start + inp_shape

            slc_tuple = []
            for a in range(-1, -(len(inp_shape) + 1), -1):
                if inp_shape[a] == 1:
                    inp_slc_start[a] = 0
                    inp_slc_stop[a] = 1
                else:
                    inp_slc_start[a] = out_slice[a].start
                    inp_slc_stop[a] = out_slice[a].stop
                slc = slice(inp_slc_start[a], inp_slc_stop[a])
                slc_tuple.insert(0, slc)
            slc_tuple = tuple(slc_tuple)
        slice_list.append(slc_tuple)

    return tuple(slice_list)


def convert_to_array_like(inputs):
    """
    Uses :obj:`numpy.asanyarray` to convert input ufunc arguments
    to array-like objects.

    :type inputs: sequence of :obj:`object`
    :param inputs: Elements of this sequence which to not have both :samp:`"shape"`
       and :samp:`"ndim"` attributes are converted to a new object
       using :obj:`numpy.asanyarray`.
    :rtype: sequence of :obj:`object`
    :return: Sequence where elements of :samp:`{inputs}` have been converted to array-like objects.

    Example::
       >>> import numpy as np
       >>> inputs = (np.array([1, 2, 3, 4], dtype="uint8"), 32.0, [[1, 2], [3, 4], [5, 6]])
       >>> convert_to_array_like(inputs)
       (array([1, 2, 3, 4], dtype=uint8), array(32.0), array([[1, 2],
              [3, 4],
              [5, 6]]))
       >>> converted = convert_to_array_like(inputs)
       >>> converted[0] is inputs[0]
       True
       >>> converted[1] is inputs[1]
       False
       >>> converted[2] is inputs[2]
       False
    """
    return \
        tuple(
            input
            if hasattr(input, "shape") and hasattr(input, "ndim") else _np.asanyarray(input)
            for input in inputs
        )


def check_equivalent_inter_locale_comms(
    gndarrays,
    equivalent_compare=(_mpi.IDENT, _mpi.CONGRUENT)
):
    """
    Checks that all the :obj:`mpi_array.globale.gndarray` elements
    of :samp:`{gndarrays}` have equivalent inter-locale communicators.

    :raises ValueError: if the arrays do not have equivalent inter-locale communicators.
    """
    if (gndarrays is not None) and (len(gndarrays) > 0):
        inter_locale_comm0 = gndarrays[0].locale_comms.inter_locale_comm
        for c in (gndary.locale_comms.inter_locale_comm for gndary in gndarrays[1:]):
            if (
                (
                    (c == _mpi.COMM_NULL)
                    and
                    (inter_locale_comm0 != _mpi.COMM_NULL)
                )
                or
                (
                    (c != _mpi.COMM_NULL)
                    and
                    (inter_locale_comm0 == _mpi.COMM_NULL)
                )
                or
                _mpi.Comm.Compare(inter_locale_comm0, c) not in equivalent_compare
            ):
                raise ValueError(
                    (
                        "Got inter_locale_comm=%s (name=%s) non-congruent with "
                        +
                        " inter_locale_comm=%s (name=%s)."
                    )
                    %
                    (
                        inter_locale_comm0,
                        inter_locale_comm0.name if inter_locale_comm0 != _mpi.COMM_NULL else "",
                        c,
                        c.name if c != _mpi.COMM_NULL else ""
                    )
                )


class GndarrayArrayUfuncExecutor(object):

    """
    Instances execute a ufunc for a :obj:`mpi_array.globale.gndarray`.
    Takes care of creating outputs, remote fetching of required parts of inputs
    and forwarding call to :obj:`numpy.ufunc` instance to perform
    the computation on the locale :obj:`numpy.ndarray` instances.
    """

    def __init__(self, array_like_obj, ufunc, method, *inputs, **kwargs):
        """
        Initialise.

        :type array_like_obj: :obj:`mpi_array.globale.gndarray`
        :param array_like_obj: The :obj:`mpi_array.globale.gndarray` which
           triggered the :samp:`__array_ufunc__` call.
        :type ufunc: :obj:`numpy.ufunc`
        :param ufunc: The ufunc to be executed.
        :type method: :obj:`str`
        :param method: The name of the method of :samp:`{ufunc}` which is
           to be executed.
        :type inputs: array like
        :param inputs: The ufunc inputs.
        :type kwargs: keyword args
        :param kwargs: The ufunc keyword arguments.
        """
        self._array_like_obj = array_like_obj
        self._ufunc = ufunc
        self._method = method
        self._inputs = convert_to_array_like(inputs)
        self._kwargs = kwargs
        self._outputs = None
        if "out" in self._kwargs.keys():
            self._outputs = self._kwargs["out"]
        self._casting = None
        if "casting" in self._kwargs.keys():
            self._casting = self._kwargs["casting"]
        else:
            self._casting = "same_kind"

    @property
    def array_like_obj(self):
        """
        The :obj:`mpi_array.globale.gndarray` object which triggered the
        construction of this :obj:`GndarrayArrayUfuncExecutor` object.
        """
        return self._array_like_obj

    @property
    def peer_comm(self):
        """
        The peer :obj:`mpi4py.MPI.Comm` communicator.
        """
        return self._array_like_obj.locale_comms.peer_comm

    @property
    def intra_locale_comm(self):
        """
        The intra-locale :obj:`mpi4py.MPI.Comm` communicator.
        """
        return self._array_like_obj.locale_comms.intra_locale_comm

    @property
    def inter_locale_comm(self):
        """
        The inter-locale :obj:`mpi4py.MPI.Comm` communicator.
        """
        return self._array_like_obj.locale_comms.inter_locale_comm

    @property
    def ufunc(self):
        """
        The :obj:`numpy.ufunc` to be executed.
        """
        return self._ufunc

    @property
    def outputs(self):
        """
        The ufunc :obj:`mpi_array.globale.gndarray` output arrays.
        """
        return self._outputs

    @property
    def inputs(self):
        """
        The sequence of ufunc inputs.
        """
        return self._inputs

    @property
    def casting(self):
        """
        A :obj:`str` indicating the casting mode.
        """
        return self._casting

    @property
    def method(self):
        """
        A :obj:`str` indicating the method of the :attr:`ufunc` to be executed.
        """
        return self._method

    def get_inputs_shapes(self):
        """
        Returns a *shape* :obj:`tuple` for each element of :attr:`inputs`.

        :rtype: :obj:`tuple`
        :return: Shape of each ufunc input.
        """
        return \
            tuple(
                input.shape
                if hasattr(input, "shape") else
                _np.asarray(input).shape
                for input in self._inputs
            )

    def get_best_match_input(self, result_shape):
        """
        Returns the element of :attr:`inputs` whose globale shape
        best matches :samp:`{result_shape}`.

        :rtype: :samp:`None` or :obj:`mpi_array.globale.gndarray`.
        :return: The input array whose shape matches :samp:`{result_shape}`,
           or :samp:`None` if none of the inputs are a good match.
        """
        best_input = None
        result_shape = _np.array(result_shape, dtype="int64")
        input_shapes = self.get_inputs_shapes()
        are_same_shape = \
            _np.array(
                tuple(
                    (len(result_shape) == len(in_shape)) and _np.all(result_shape == in_shape)
                    for in_shape in input_shapes
                )
            )
        if _np.any(are_same_shape):
            best_input = self._inputs[_np.where(are_same_shape)[0][0]]
        else:
            input_shapes = \
                _np.array(
                    tuple(
                        _np.array(shape_extend_dims(len(result_shape), in_shape))
                        for in_shape in input_shapes
                    ),
                    dtype="int64"
                )
            d = input_shapes - result_shape
            d *= d
            d = d.sum(axis=1)
            best_input = self._inputs[_np.argmin(d)]

        return best_input

    def create_outputs(self, outputs, result_shape, result_types):
        """
        Returns list of output :obj:`mpi_array.globale.gndarray` instances.

        :type outputs: :samp:`None` or :obj:`tuple` of :obj:`mpi_array.globale.gndarray`
        :param outputs: Output arrays passed in as the :samp:`out` argument
           of the :obj:`numpy.ufunc`.
        :type result_shape: sequence of :obj:`int`
        :param result_shape: The shape of all output arrays.
        :type result_types: sequence of :samp:`numpy.dtype`
        :param result_types: The :samp:`dtype` of each output array. Note
            that this is the list for all outputs including any
            in the :samp:`outputs` argument. This determines the
            number of output arrays.
        :rtype: :obj:`list` of :obj:`mpi_array.globale.gndarray`
        :return: A list of length :samp:`len(result_types)` elements,
           each element is a :obj:`mpi_array.globale.gndarray`.
        """

        template_output_gary = None
        if (outputs is not None) and (len(outputs) > 0):
            check_equivalent_inter_locale_comms(outputs)
            template_output_gary = outputs[-1]
        else:
            best_match_input = self.get_best_match_input(result_shape)
            comms_distrib = None
            if best_match_input is not None:
                comms_distrib = \
                    _comms.reshape_comms_distribution(
                        best_match_input.comms_and_distrib,
                        result_shape
                    )
            if comms_distrib is not None:
                template_output_gary = \
                    _globale_creation.empty(
                        result_shape,
                        comms_and_distrib=comms_distrib,
                        dtype=result_types[0]
                    )
            else:
                template_output_gary = \
                    _globale_creation.empty(
                        result_shape,
                        dtype=result_types[0],
                        peer_comm=self.peer_comm,
                        intra_locale_comm=self.intra_locale_comm,
                        inter_locale_comm=self.inter_locale_comm
                    )
            outputs = (template_output_gary,)
        outputs = \
            (
                outputs
                +
                tuple(
                    _globale_creation.empty_like(template_output_gary, dtype=result_types[i])
                    for i in range(len(outputs), len(result_types))
                )
            )

        return outputs

    def get_input_extents(self, locale_info):
        """
        Returns tuple of :samp:`(locale_extent, globale_extent)` pairs,
        one for each of the :attr:`inputs`.

        :type locale_info: :obj:`mpi_array.comms.ThisLocaleInfo`
        :param locale_info: The rank info required for constructing
            a :obj:`mpi_array.distribution.LocaleExtent` instance
            for :samp:`input` types which are not :obj:`mpi_array.globale.gndarray`.
        :rtype: :obj:`tuple`
        :return: Pairs which indicate the locale extent of the ufunc :attr:`inputs`.

        .. seealso:: :func:`get_extents`
        """
        return \
            tuple(
                get_extents(inp, locale_info) for inp in self.inputs
            )

    def get_numpy_ufunc_peer_rank_inputs_outputs(self, gndarray_outputs):
        """
        Returns two element tuple of :samp:`(input_arrays, output_arrays)` which
        are to be passed to the :obj:`numpy.ufunc` object :attr:`ufunc`.

        :type gndarray_outputs: sequence of :obj:`mpi_array.globale.gndarray`
        :param gndarray_outputs: The output arrays. All arrays should be the
           same shape and same distribution.
        :rtype: :samp:`None` or :obj:`tuple`
        :return: A tuple :samp:`(input_arrays, output_arrays)` of inputs and
           outputs which are to be passed to :obj:`numpy.ufunc` call.
           Returns :samp:`None` if the output locale extents are empty (i.e. no
           array elements to compute on this locale).
        """
        # First fetch/slice the parts of the input required for the locale extent
        out_gndarray = gndarray_outputs[0]
        out_globale_extent = out_gndarray.distribution.globale_extent
        out_locale_extent = out_gndarray.lndarray_proxy.locale_extent
        ret = None
        if _np.product(out_locale_extent.shape_n) > 0:
            inp_locale_extents = \
                self.get_input_extents(out_gndarray.comms_and_distrib.this_locale)
            inp_locale_slices = \
                calc_matching_locale_slices(
                    out_locale_extent,
                    out_globale_extent,
                    inp_locale_extents
                )

            inp_locale_arys = [None, ] * len(self.inputs)
            for i in range(len(self.inputs)):
                input = self.inputs[i]
                slice_tuple = inp_locale_slices[i]
                if slice_tuple is not None:
                    if hasattr(input, "locale_get"):
                        # is a gndarray
                        inp_locale_arys[i] = input.locale_get(slice_tuple)
                    else:
                        # is a numpy array (or similar)
                        inp_locale_arys[i] = input[slice_tuple]
                else:
                    # is a scalar
                    inp_locale_arys[i] = input

            # Now slice the locale input arrays to match the peer-rank portions of the output.
            out_peer_rank_slice = out_gndarray.lndarray_proxy.intra_partition.rank_view_slice_n
            out_peer_rank_slice = out_locale_extent.locale_to_globale_slice_h(out_peer_rank_slice)
            out_peer_rank_slice = out_locale_extent.globale_to_locale_slice_n(out_peer_rank_slice)

            inp_peer_rank_slices = calc_matching_peer_rank_slices(
                out_peer_rank_slice, inp_locale_arys)

            inp_peer_rank_arys = [None, ] * len(inp_locale_arys)
            for i in range(len(inp_locale_arys)):
                input = inp_locale_arys[i]
                slice_tuple = inp_peer_rank_slices[i]
                if slice_tuple is not None:
                    # is a numpy array (or similar)
                    inp_peer_rank_arys[i] = input[slice_tuple]
                else:
                    # is a scalar
                    inp_peer_rank_arys[i] = input

            ret = \
                (
                    tuple(inp_peer_rank_arys),
                    tuple(
                        out_gndarray.view_n[out_peer_rank_slice]
                        for out_gndarray in gndarray_outputs
                    )
                )
        return ret

    def need_remote_data(self, gndarray_outputs):
        """
        Returns :samp:`True` if any locale needs to fetch remote
        input data in order to compute the all elements of the
        outputs :samp:`{gndarray_outputs}`.

        :type gndarray_outputs: sequence of :obj:`mpi_array.globale.gndarray`
        :param gndarray_outputs: Check whether any of the locales require remote
           data in order to compute these outputs.
        :rtype: :obj:`bool`
        :return: :samp:`True` if remote fetch of input data is required
           in order to compute ufunc for the given outputs.
        """
        out_gndary = gndarray_outputs[0]
        need_remote = False
        if out_gndary.locale_comms.inter_locale_comm != _mpi.COMM_NULL:
            START_STR = LocaleExtent.START_N_STR
            STOP_STR = LocaleExtent.STOP_N_STR
            gndarray_inputs = \
                tuple(
                    input for input in self.inputs
                    if hasattr(input, "distribution") and hasattr(input, "locale_comms")
                )
            out_s_ext = out_gndary.distribution.struct_locale_extents
            for inp_gndary in gndarray_inputs:
                need_remote = \
                    (
                        _mpi.Comm.Compare(
                            out_gndary.locale_comms.inter_locale_comm,
                            inp_gndary.locale_comms.inter_locale_comm
                        )
                        ==
                        _mpi.UNEQUAL
                    )
                if not need_remote:
                    # first make sure that the inter_locale_comm is compatible
                    # between input and output
                    translated_ranks = \
                        _mpi.Group.Translate_ranks(
                            out_gndary.locale_comms.inter_locale_comm.group,
                            _np.arange(out_gndary.locale_comms.inter_locale_comm.group.size),
                            inp_gndary.locale_comms.inter_locale_comm.group
                        )
                    inp_s_ext = \
                        inp_gndary.distribution.struct_locale_extents[_np.asarray(translated_ranks)]

                    # Now check that the output locale extent is contained
                    # within the input locale extent.
                    # Dimension of input can be smaller than the output
                    # because of broadcasting rules.
                    need_remote = True
                    not_out_empty = \
                        _np.product(out_s_ext[STOP_STR] - out_s_ext[START_STR], axis=1) > 0

                    ndim = inp_gndary.ndim

                    beyond_out_extent = \
                        _np.logical_or.reduce(
                            (out_s_ext[START_STR][:, -ndim:] < inp_s_ext[START_STR])
                            |
                            (out_s_ext[STOP_STR][:, -ndim:] <= inp_s_ext[START_STR])
                            |
                            (out_s_ext[START_STR][:, -ndim:] >= inp_s_ext[STOP_STR])
                            |
                            (out_s_ext[STOP_STR][:, -ndim:] > inp_s_ext[STOP_STR]),
                            axis=1
                        )

                    need_remote = \
                        _np.any(
                            not_out_empty
                            &
                            beyond_out_extent
                        )

                if need_remote:
                    break
        # All ranks in the locale need to know the result, broadcast.
        need_remote = out_gndary.locale_comms.intra_locale_comm.bcast(need_remote, 0)

        return need_remote

    def execute___call__(self):
        """
        """
        from .globale import gndarray as _gndarray

        # Calculate the shape of the output arrays.
        result_shape = broadcast_shape(*(self.get_inputs_shapes()))
        self.array_like_obj.rank_logger.debug("result_shape=%s", result_shape)

        # Calculate the result dtype for each output array
        result_types = ufunc_result_type(self.ufunc.types, self.inputs, self.outputs, self.casting)
        self.array_like_obj.rank_logger.debug("result_types=%s", result_types)

        # Create the output gndarray instances
        gndarray_outputs = self.create_outputs(self.outputs, result_shape, result_types)
        self.array_like_obj.rank_logger.debug(
            "output shapes=%s", [o.shape for o in gndarray_outputs]
        )

        # Check whether remote fetch of data is needed
        # for any locale before calling this barrier. If all locales
        # have local data then this barrier isn't be necessary.
        # Otherwise, we have to sync to make sure that remote ranks have
        # finished writing data before starting to fetch it.
        if self.need_remote_data(gndarray_outputs):
            for i in self.inputs:
                if isinstance(i, _gndarray):
                    i.initialise_windows()
            gndarray_outputs[0].inter_locale_barrier()

        # Fetch the peer-rank sub-arrays of the input arrays needed
        # to calculate the corresponding sub-array of the outputs.
        np_ufunc_inputs_and_outputs = \
            self.get_numpy_ufunc_peer_rank_inputs_outputs(gndarray_outputs)

        if np_ufunc_inputs_and_outputs is not None:
            np_ufunc_inputs, np_ufunc_outputs = np_ufunc_inputs_and_outputs

            # Call the self.ufunc.__call__ method to perform the computation
            # in the sub-arrays
            kwargs = dict()
            kwargs.update(self._kwargs)
            kwargs["out"] = np_ufunc_outputs
            self.array_like_obj.rank_logger.debug(
                "Calling numpy.ufunc=%s:\ninputs=%s\noutputs=%s",
                self.ufunc, np_ufunc_inputs, kwargs["out"]
            )
            self.ufunc.__call__(*np_ufunc_inputs, **kwargs)
            self.array_like_obj.rank_logger.debug(
                "Finished numpy.ufunc=%s:\noutputs=%s",
                self.ufunc,
                kwargs["out"]
            )
        else:
            self.array_like_obj.rank_logger.debug(
                "Locale output extent is empty, skipping call to self.ufunc=%s:\nOutput extent=%s",
                self.ufunc,
                gndarray_outputs[0].lndarray_proxy.locale_extent
            )

        gndarray_outputs[0].intra_locale_barrier()

        # return the outputs
        if len(gndarray_outputs) == 1:
            gndarray_outputs = gndarray_outputs[0]
        return gndarray_outputs

    def execute_accumulate(self):
        """
        Not implemented.
        """
        return NotImplemented

    def execute_reduce(self):
        """
        Not implemented.
        """
        return NotImplemented

    def execute_reduceat(self):
        """
        Not implemented.
        """
        return NotImplemented

    def execute_at(self):
        """
        Not implemented.
        """
        return NotImplemented

    def execute_outer(self):
        """
        Not implemented.
        """
        return NotImplemented

    def execute(self):
        """
        Perform the ufunc operation. Call is forwarded to one
        of: :meth:`execute___call__`, :meth:`execute_accumulate`, :meth:`execute_at`
        , :meth:`execute_outer`, :meth:`execute_reduce` or :meth:`execute_reduceat`.
        """
        return getattr(self, "execute_" + self.method)()


#: Factory for generating instance of :obj:`GndarrayArrayUfuncExecutor`.
gndarray_ufunc_executor_factory = GndarrayArrayUfuncExecutor


def gndarray_array_ufunc(array_like_obj, ufunc, method, *inputs, **kwargs):
    """
    The implementation for  :meth:`mpi_array.globale.gndarray.__array_ufunc__`.
    """
    ufunc_executor = \
        gndarray_ufunc_executor_factory(
            array_like_obj,
            ufunc,
            method,
            *inputs,
            **kwargs
        )

    return ufunc_executor.execute()


def set_numpy_ufuncs_as_module_attr(set_attr_module, search_module):
    """
    Finds all :obj:`numpy.ufunc` attributes in the :samp:`{search_module}` :obj:`module`
    and sets corresponding attributes of :samp:`{set_attr_module}` :obj:`module`.

    :type set_attr_module: :obj:`module`
    :param set_attr_module: Set ufunc attributes of this module to those found
       in the :samp:`{search_module}` module
    :type search_module: :obj:`module`
    :param search_module: Find :obj:`numpy.ufunc` attributes in this module.

    """
    for attr in dir(search_module):
        numpy_attr_value = getattr(search_module, attr)
        if isinstance(numpy_attr_value, _np.ufunc):
            setattr(set_attr_module, attr, numpy_attr_value)


set_numpy_ufuncs_as_module_attr(_sys.modules[__name__], _np)

__all__ = [s for s in dir() if not s.startswith('_')]
