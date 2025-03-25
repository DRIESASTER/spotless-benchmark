#!/usr/bin/env python3

##Here is where i tried a plethora of things trying to make it work to train in batches but kept running into issues i'll add some of the attempts in there simply to show that i've been busy

import argparse as arp
import os


def main():
    ##### PARSING COMMAND LINE ARGUMENTS #####
    prs = arp.ArgumentParser()

    prs.add_argument('sp_data_path', type=str, help='path of spatial data')

    prs.add_argument('model_path', type=str, help='path to regression model')

    prs.add_argument('cuda_device', type=str, help="index of cuda device ID or cpu")

    prs.add_argument('-o', '--out_dir', default=os.getcwd(),
                     type=str, help='model and proportions output directory')

    prs.add_argument('-e', '--epochs', default=30000, type=int, help="number of epochs to fit the model")

    prs.add_argument('-p', '--posterior_sampling', default=1000, type=int,
                     help="number of samples to take from the posterior distribution")

    prs.add_argument('-n', '--n_cells_per_location', default=8, type=int, help="estimated number of cells per spot")

    prs.add_argument('-d', '--detection_alpha', default=200, type=int,
                     help="within-experiment variation in RNA detection sensitivity")

    args = prs.parse_args()

    cuda_device = args.cuda_device
    sp_data_path = args.sp_data_path
    output_folder = args.out_dir

    assert (cuda_device.isdigit() or cuda_device == "cpu"), "invalid device input"

    print("Parameters\n==========")
    print("Detection alpha: {}\nCells per location: {}".format(args.detection_alpha, args.n_cells_per_location))
    print("Training epochs: {}\nPosterior sampling: {}".format(args.epochs, args.posterior_sampling))
    print("==========")

    ##### MAIN PART #####
    if cuda_device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


    import scanpy as sc
    import numpy as np

    import cell2location
    import scvi

    from matplotlib import rcParams
    rcParams['pdf.fonttype'] = 42

    # silence scanpy that prints a lot of warnings
    import warnings
    warnings.filterwarnings('ignore')

    print("Reading in spatial data from " + sp_data_path + "...")
    adata = sc.read_h5ad(sp_data_path)
    adata.var['SYMBOL'] = adata.var_names

    # mitochondria-encoded (MT) genes should be removed for spatial mapping
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata = adata[:, ~adata.var['mt'].values]

    adata_vis = adata.copy()
    adata_vis.raw = adata_vis

    print("Reading in the model...")
    adata_scrna_raw = sc.read(args.model_path)

    # Export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_scrna_raw.varm.keys():
        inf_aver = adata_scrna_raw.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                                    for i in
                                                                    adata_scrna_raw.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_scrna_raw.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_scrna_raw.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_scrna_raw.uns['mod']['factor_names']

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    #new
    chunk_size = 72_000
    chunks = [i for i in range(int(np.ceil(adata_vis.n_obs / chunk_size)))]
    #prints chuncks
    print("splitting into: "+str(chunks))
    print("Columns in adata_vis.obs:", adata_vis.obs.columns.tolist())
    adata_vis.obs['training_batch'] = 0
    for sample in adata_vis.obs['orig.ident'].unique():
        ind = adata_vis.obs['orig.ident'].isin([sample])
        adata_vis.obs.loc[ind, 'training_batch'] = np.random.choice(
            chunks, size=ind.sum(), replace=True, p=None
        )

    adata_vis_full = adata_vis.copy()
    for k in ['means', 'stds', 'q05', 'q95']:
        adata_vis_full.obsm[f"{k}_cell_abundance_w_sf"] = np.zeros((adata_vis_full.n_obs, inf_aver.shape[1]))

    adata_vis.obs['training_batch'].value_counts()
    print("Training batches NEW:")
    print(adata_vis.obs['training_batch'].value_counts())

    seed = 0
    scvi.settings.seed = seed
    np.random.seed(seed)

    # submit this chunk as separate jobs
    for batch in adata_vis.obs['training_batch'].unique():
        # create and train the model
        scvi_run_name = f'fit_model_batch{batch}_seed{seed}'
        print(scvi_run_name)

        training_batch_index = adata_vis_full.obs['training_batch'].isin([batch])
        adata_vis = adata_vis_full[training_batch_index, :].copy()

        # prepare anndata for scVI model
        cell2location.models.Cell2location.setup_anndata(adata_vis, batch_key="orig.ident")

        # Step 2: Instantiate the Cell2location model.
        mod = cell2location.models.Cell2location(
            adata_vis,
            cell_state_df=inf_aver,  # reference signatures from your scRNA-seq data
            N_cells_per_location=args.n_cells_per_location,
            detection_alpha=args.detection_alpha
        )

        # Now you can train the model.
        mod.train(max_epochs=args.epochs, batch_size=None, train_size=1, use_gpu=cuda_device.isdigit())

        # export posterior
        # In this section, we export the estimated cell abundance (summary of the posterior distribution).

        adata_vis = mod.export_posterior(
    		adata_vis,
    		sample_kwargs={
  	     	'num_samples': args.posterior_sampling,
        	'batch_size': mod.adata.n_obs,
        	'use_gpu': cuda_device.isdigit()
    	},
    	add_to_obsm=['q05']
	)

    for k in adata_vis_full.obsm.keys():
        adata_vis_full.obsm[k][training_batch_index, :] = adata_vis.obsm[k].copy()
    adata_vis_full.uns[f'mod_{batch}'] = adata_vis.uns['mod'].copy()




        # Create and train the model
        print(f"Creating and training model for batch {batch}...")
        mod = cell2location.models.Cell2location(
            adata_vis_batch, cell_state_df=inf_aver,
            N_cells_per_location=args.n_cells_per_location,
            detection_alpha=args.detection_alpha
        )
        mod.train(
            max_epochs=args.epochs,
            batch_size=None,  # Use full data for training
            train_size=1,
            use_gpu=cuda_device.isdigit()
        )

        # Export the posterior distribution for the current batch
        print(f"Exporting posterior distribution for batch {batch}...")
        adata_vis_batch = mod.export_posterior(
            adata_vis_batch, sample_kwargs={
                'num_samples': args.posterior_sampling,
                'batch_size': mod.adata.n_obs,
                'use_gpu': cuda_device.isdigit()
            }
        )

        # Aggregate results into the full dataset
        print(f"Aggregating results for batch {batch}...")
        for k in adata_vis_batch.obsm.keys():
            if k in adata_vis_full.obsm:
                adata_vis_full.obsm[k][training_batch_index, :] = adata_vis_batch.obsm[k].copy()

        # Save the model and results for the current batch
        print(f"Saving model and results for batch {batch}...")
        mod.save(os.path.join(output_folder, f'mod_batch_{batch}'), overwrite=True)
        adata_vis_batch.write(os.path.join(output_folder, f'sp_batch_{batch}.h5ad'))

    # Save the full aggregated dataset
    print("Saving the full aggregated dataset...")
    adata_vis_full.write(os.path.join(output_folder, 'sp_full.h5ad'))

    # Export proportion file
    print("Exporting proportion file...")
    props = adata_vis_full.obsm['q05_cell_abundance_w_sf']
    print("Columns before renaming:", props.columns)
    props = props.rename(columns={x: x.replace("q05_cell_abundance_w_sf_", "") for x in props.columns})
    print("Columns after renaming:", props.columns)
    props = props.div(props.sum(axis=1), axis='index')
    props.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")
    print("Proportions file saved successfully.")











    # # prepare anndata for cell2location model
    # scvi.data.setup_anndata(adata=adata_vis)
    #
    # # Create and train the model
    # mod = cell2location.models.Cell2location(
    #     adata_vis, cell_state_df=inf_aver,
    #     # the expected average cell abundance: tissue-dependent
    #     # hyper-prior which can be estimated from paired histology:
    #     N_cells_per_location=args.n_cells_per_location,
    #     # hyperparameter controlling normalisation of
    #     # within-experiment variation in RNA detection (using default here):
    #     detection_alpha=args.detection_alpha
    # )
    #
    # mod.train(max_epochs=args.epochs,
    #           # train using full data (batch_size=None)
    #           batch_size=None,
    #           # use all data points in training because
    #           # we need to estimate cell abundance at all locations
    #           train_size=1,
    #           use_gpu=cuda_device.isdigit())
    #
    # # Export the estimated cell abundance (summary of the posterior distribution).
    # adata_vis = mod.export_posterior(
    #     adata_vis, sample_kwargs={'num_samples': args.posterior_sampling,
    #                               'batch_size': mod.adata.n_obs, 'use_gpu': cuda_device.isdigit()}
    # )
    #
    # # Save model and anndata object with results
    # mod.save(output_folder, overwrite=True)
    # adata_vis.write(os.path.join(output_folder, 'sp.h5ad'))
    #
    # # Export proportion file, but first rename columns and divide by rowSums
    # props = adata_vis.obsm['q05_cell_abundance_w_sf']
    # props = props.rename(columns={x: x.replace("q05cell_abundance_w_sf_", "") for x in props.columns})
    # props = props.div(props.sum(axis=1), axis='index')
    # props.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")

    # df = pd.DataFrame(data=np.random.normal(size=(10,10)),
    #                     index=["row"+str(i) for i in range(10)],
    #                     columns=["col"+str(i) for i in range(10)])
    # df.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")


mod.train(max_epochs=args.epochs,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=cuda_device.isdigit())

    # Export the estimated cell abundance (summary of the posterior distribution).
    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': args.posterior_sampling,
                                  'batch_size': mod.adata.n_obs, 'use_gpu': cuda_device.isdigit()}
    )

    # Save model and anndata object with results
    mod.save(output_folder, overwrite=True)
    adata_vis.write(os.path.join(output_folder, 'sp.h5ad'))

    # Export proportion file, but first rename columns and divide by rowSums
    props = adata_vis.obsm['q05_cell_abundance_w_sf']
    props = props.rename(columns={x: x.replace("q05cell_abundance_w_sf_", "") for x in props.columns})
    props = props.div(props.sum(axis=1), axis='index')
    props.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")

    # df = pd.DataFrame(data=np.random.normal(size=(10,10)),
    #                     index=["row"+str(i) for i in range(10)],
    #                     columns=["col"+str(i) for i in range(10)])
    # df.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")

        # Subset the data for the current batch
        print(f"Subsetting data for batch {batch}...")
        training_batch_index = adata_vis_full.obs['training_batch'] == batch
        adata_vis_batch = adata_vis_full[training_batch_index, :].copy()

        # Prepare anndata for the Cell2location model
        print(f"Preparing anndata for batch {batch}...")
        cell2location.models.Cell2location.setup_anndata(
            adata=adata_vis_batch, batch_key="training_batch"  # Use 'training_batch' as the batch key
        )

        # Create and train the model
        print(f"Creating and training model for batch {batch}...")
        mod = cell2location.models.Cell2location(
            adata_vis_batch, cell_state_df=inf_aver,
            N_cells_per_location=args.n_cells_per_location,
            detection_alpha=args.detection_alpha
        )
        mod.train(
            max_epochs=args.epochs,
            batch_size=None,  # Use full data for training
            train_size=1,
            use_gpu=cuda_device.isdigit()
        )

        # Export the posterior distribution for the current batch
        print(f"Exporting posterior distribution for batch {batch}...")
        adata_vis_batch = mod.export_posterior(
            adata_vis_batch, sample_kwargs={
                'num_samples': args.posterior_sampling,
                'batch_size': mod.adata.n_obs,
                'use_gpu': cuda_device.isdigit()
            }
        )

        # Aggregate results into the full dataset
        print(f"Aggregating results for batch {batch}...")
        for k in adata_vis_batch.obsm.keys():
            if k in adata_vis_full.obsm:
                adata_vis_full.obsm[k][training_batch_index, :] = adata_vis_batch.obsm[k].copy()

        # Save the model and results for the current batch
        print(f"Saving model and results for batch {batch}...")
        mod.save(os.path.join(output_folder, f'mod_batch_{batch}'), overwrite=True)
        adata_vis_batch.write(os.path.join(output_folder, f'sp_batch_{batch}.h5ad'))

    # Save the full aggregated dataset
    print("Saving the full aggregated dataset...")
    adata_vis_full.write(os.path.join(output_folder, 'sp_full.h5ad'))

    # Export proportion file
    print("Exporting proportion file...")
    props = adata_vis_full.obsm['q05_cell_abundance_w_sf']
    print("Columns before renaming:", props.columns)
    props = props.rename(columns={x: x.replace("q05_cell_abundance_w_sf_", "") for x in props.columns})
    print("Columns after renaming:", props.columns)
    props = props.div(props.sum(axis=1), axis='index')
    props.to_csv(os.path.join(output_folder, 'proportions.tsv'), sep="\t")
    print("Proportions file saved successfully.")


if __name__ == '__main__':


if __name__ == '__main__':
    main()
