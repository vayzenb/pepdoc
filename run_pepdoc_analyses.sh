
echo "Running erp"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/eeg/extract_erp.py

echo "Running decoding"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/eeg/pepdoc_decoding.py

echo "Running concatenating timeseries"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/eeg/concat_timeseries.py

echo "Creating RDMs"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/eeg/create_rdms.py

echo "Run tga"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/analysis/correlate_rdms.py

echo "run PCA"
python /user_data/vayzenbe/GitHub_Repos/pepdoc/analysis/channel_pca.py