protVI
======
A deep generative model for single-cell MS proteomics data.

.. code-block:: python

    from protvi.model import PROTVI

    PROTVI.setup_anndata(adata, batch_key="batch_id")

    model = PROTVI(adata, n_hidden=32)
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
