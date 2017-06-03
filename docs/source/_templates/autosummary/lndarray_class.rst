..
   Special template for mpi_array.local.lndarray to avoid numpydoc
   documentation style warnings/errors from numpy.ndarray inheritance.

{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/methods

      ~lndarray.__new__
      ~lndarray.__array_finalize__

   {% endblock %}

   {% block attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: generated/attribs
      
      ~lndarray.md
      ~lndarray.decomp
      ~lndarray.rank_view_h
      ~lndarray.rank_view_n

   {% endblock %}
