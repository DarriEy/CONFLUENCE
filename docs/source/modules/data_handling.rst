Data Handling Module
====================

The data handling module manages all data acquisition, preprocessing, and management tasks.

Components
----------

Data Acquisition
~~~~~~~~~~~~~~~~

.. autoclass:: utils.data.data_utils.DataAcquisitionProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Attribute Processing
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: utils.data.attribute_processing_util.attributeProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Processing Streamflow Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils.data.data_utils import ObservedDataProcessor
   
   processor = ObservedDataProcessor(config, logger)
   processor.process_streamflow_data()

Acquiring Forcing Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils.data.data_utils import datatoolRunner
   
   dr = datatoolRunner(config, logger)
   dr.execute_datatool_command(command)

See Also
--------

- :doc:`geospatial` for spatial data handling
- :doc:`models` for model-specific data processing
