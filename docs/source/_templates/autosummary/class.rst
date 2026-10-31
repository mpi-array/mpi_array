{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :special-members: __enter__, __exit__, __del__, __getitem__, __setitem__, __array_ufunc__

   {% set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list %}
   {% if own_methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% set own_attributes = attributes | reject("in", inherited_members) | list %}
   {% if own_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in own_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
