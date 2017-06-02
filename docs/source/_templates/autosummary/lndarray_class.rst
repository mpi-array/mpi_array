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

   {% endblock %}
