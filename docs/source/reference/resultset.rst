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
   ResultSet.set
   ResultSet.get_nth_last_result
   ResultSet.get_nth_last_params
   ResultSet.is_computed


Input-output
------------

.. autosummary::
   :toctree: api/

   ResultSet.save
   ResultSet.load
   ResultSet.get_save_path
   ResultSet.save_path_fmt
   ResultSet.set_save_fun


Timing
------

.. autosummary::
   :toctree: api/

   ResultSet.get_time
   ResultSet.get_nth_last_time
   ResultSet.get_timing_stats
   ResultSet.rank_longest_to_compute


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

   ResultSet.parameters
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

   ResultSet.results
   ResultSet.set_compute_fun
   ResultSet.set_save_fun
   ResultSet.add_results


Create other sets
-----------------
.. autosummary::
   :toctree: api/

   ResultSet.get_subspace_res
