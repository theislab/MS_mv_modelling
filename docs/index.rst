protVI
======
A deep generative model for single-cell mass spectrometry (MS)-based proteomics data, supporting:

- Missing value imputation
- Batch correction
- Differential abundance analysis

.. note::
    The model is still in active development.

Quick start
-----------
.. code-block:: python

    from protvi.model import PROTVI

    PROTVI.setup_anndata(adata, batch_key="batch_id")

    model = PROTVI(adata)
    model.train()


.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   tutorials/index
   api
   references



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
