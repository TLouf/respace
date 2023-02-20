==========
Result Set
==========

.. currentmodule:: respace

Constructor
-----------

.. autosummary::
   :toctree: api/

   ResultSet


Usual workflow
--------------

.. autosummary::
   :toctree: api/

   ResultSet.compute
   ResultSet.get
   ResultSet.save
   ResultSet.set
   ResultSet.load
   ResultSet.get_nth_last_computed
   ResultSet.get_nth_last_details
   ResultSet.get_save_path
   ResultSet.is_computed


Accessing the parameter space
-----------------------------
.. autosummary::
   :toctree: api/

   ResultSet.__getitem__
   ResultSet.param_space
   ResultSet.coords
   ResultSet.attrs
   ResultSet.populated_mask
   ResultSet.populated_space


Parameters manipulation
-----------------------

.. autosummary::
   :toctree: api/

   ResultSet.params_defaults
   ResultSet.params_values
   ResultSet.coords
   ResultSet.fill_with_defaults
   ResultSet.add_param_values
   ResultSet.add_params


Results manipulation
--------------------

.. autosummary::
   :toctree: api/

   ResultSet.results_metadata
   ResultSet.set_compute_fun
   ResultSet.set_save_fun
   ResultSet.add_results


Create other sets
-----------------
.. autosummary::
   :toctree: api/

   ResultSet.get_subspace_res
