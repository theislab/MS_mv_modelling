protVI
======
A deep generative model for single-cell MS proteomics data.

Quick start
-----------
.. code-block:: python

    from protvi.model import PROTVI

    PROTVI.setup_anndata(adata, batch_key="batch_id")

    model = PROTVI(adata)
    model.train()



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   examples/index
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
