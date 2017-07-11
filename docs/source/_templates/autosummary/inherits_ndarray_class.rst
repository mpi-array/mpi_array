..
   Special template for mpi_array.locale.slndarray to avoid numpydoc
   documentation style warnings/errors from numpy.ndarray inheritance.

{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/methods

      ~{{ name }}.__new__
      ~{{ name }}.__array_finalize__

   {% endblock %}

   {% block attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: generated/attribs
      
      ~{{ name }}.md

   {% endblock %}
