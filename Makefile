mimic_download:
	python msr/bin/download_data/mimic.py download=True create_splits=True

mimic_create_representations:
	python msr/bin/create_representations/mimic.py n_jobs=1



ptbxl_download:
	python msr/bin/download_data/ptbxl.py download=True create_splits=True

ptbxl_create_representations:
	python msr/bin/create_representations/ptbxl.py n_jobs=1

ptbxl_train:
	python msr/bin/train/ptbxl.py experiment=ptbxl datamodule.representation_type="whole_signal_features"


sleep_edf_download:
	python msr/bin/download_data/sleep_edf.py download=True create_raw_csv=True create_splits=True

sleep_edf_create_representations:
	python msr/bin/create_representations/sleep_edf.py n_jobs=1
