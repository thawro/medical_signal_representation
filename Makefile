mimic_download:
	python msr/bin/download_data/mimic.py download=True create_splits=True

mimic_create_representations:
	python msr/bin/create_representations.py dataset_provider=mimic

mimic_experiments:
	python msr/bin/train_and_evaluate.py \
		experiment=mimic/lgbm,mimic/decision_tree,mimic/regression,mimic/mlp,mimic/cnn \
		representation_type=whole_signal_waveforms,whole_signal_features,agg_beat_waveforms,agg_beat_features \
		--multirun



ptbxl_download:
	python msr/bin/download_data/ptbxl.py download=True create_splits=True

ptbxl_create_representations:
	python msr/bin/create_representations.py dataset_provider=ptbxl

ptbxl_experiments:
	python msr/bin/train_and_evaluate.py \
		experiment=ptbxl/lgbm,ptbxl/decision_tree,ptbxl/regression,ptbxl/mlp,ptbxl/cnn \
		representation_type=whole_signal_waveforms,whole_signal_features,agg_beat_waveforms,agg_beat_features \
		--multirun



sleep_edf_download:
	python msr/bin/download_data/sleep_edf.py download=True create_raw_csv=True create_splits=True

sleep_edf_create_representations:
	python msr/bin/create_representations.py dataset_provider=sleep_edf batch_size=20000

sleep_edf_experiments:
	python msr/bin/train_and_evaluate.py \
		experiment=sleep_edf/lgbm,sleep_edf/decision_tree,sleep_edf/regression,sleep_edf/mlp,sleep_edf/cnn \
		representation_type=whole_signal_waveforms,whole_signal_features \
		--multirun


experiments:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/cnn representation_type=whole_signal_waveforms
	python msr/bin/train_and_evaluate.py experiment=mimic/cnn representation_type=whole_signal_waveforms
	python msr/bin/train_and_evaluate.py experiment=sleep_edf/cnn representation_type=whole_signal_waveforms
	python msr/bin/train_and_evaluate.py experiment=mimic/cnn datamodule=mimic_clean representation_type=whole_signal_waveforms
