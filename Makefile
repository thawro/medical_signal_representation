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


decision_tree_hparams:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/decision_tree \
		representation_type=agg_beat_waveforms model.max_features=0.1,0.3,0.5,0.7,0.9,1.0 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/decision_tree \
		representation_type=agg_beat_waveforms model.min_samples_split=2,4,8,16 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/decision_tree \
		representation_type=agg_beat_waveforms model.max_depth=4,16,32,64,null --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/decision_tree \
		representation_type=agg_beat_waveforms model.min_samples_leaf=1,2,4,8,16 --multirun


lgbm_hparams:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/lgbm \
		representation_type=whole_signal_waveforms model.colsample_bytree=0.1,0.3,0.5,0.7,0.9,1.0 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/lgbm \
		representation_type=whole_signal_waveforms model.max_depth=4,16,32,64,-1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/lgbm \
		representation_type=whole_signal_waveforms model.num_leaves=7,15,31,63 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/lgbm \
		representation_type=whole_signal_waveforms model.n_estimators=100,250,600,1000 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/lgbm \
		representation_type=whole_signal_waveforms model.learning_rate=0.001,0.005,0.01,0.1 --multirun

# TODO
mlp_hparams:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/mlp \
		representation_type=whole_signal_waveforms model.learning_rate=0.001,0.01,0.05,0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/mlp \
		representation_type=whole_signal_waveforms model.net.hidden_dims='[512]','[256, 512]','[512, 256]','[256, 512, 256]','[128, 256, 512, 256, 128]' --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/mlp \
		representation_type=whole_signal_waveforms model.net.dropout=0,0.1,0.2,0.3,0.4 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/mlp \
		representation_type=whole_signal_waveforms model.weight_decay=0.0001,0.001,0.01,0.1 --multirun



cnn_hparams:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.learning_rate=0.001,0.01,0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.net.conv0_kernel_size=5,7,9 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.net.conv0_channels=64,128,256 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.net.layers='[1]','[1, 1]','[1, 1, 1]','[1, 1, 1, 1]' --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.net.ff_hidden_dims='[64]','[128]','[128, 128]','[128, 64]' --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.net.ff_dropout=0,0.1,0.2,0.3 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/hparams_optimization/cnn \
		representation_type=agg_beat_features model.weight_decay=0.0001,0.001,0.01 --multirun






# TODO after all hparams optimizations
regression_experiments:
	python msr/bin/train_and_evaluate.py experiment=mimic/regression representation_type=whole_signal_waveforms
	python msr/bin/train_and_evaluate.py experiment=ptbxl/regression,sleep_edf/regression representation_type=whole_signal_waveforms \
		model.C=10 model.solver=lbfgs model.max_iter=2000 --multirun
	python msr/bin/train_and_evaluate.py experiment=mimic/regression representation_type=whole_signal_features
	python msr/bin/train_and_evaluate.py experiment=ptbxl/regression,sleep_edf/regression representation_type=whole_signal_features \
		model.C=0.001 model.solver=newton-cg model.max_iter=2000 --multirun
	python msr/bin/train_and_evaluate.py experiment=mimic/regression representation_type=agg_beat_waveforms
	python msr/bin/train_and_evaluate.py experiment=ptbxl/regression representation_type=agg_beat_waveforms \
		model.C=0.1 model.solver=sag model.max_iter=2000 --multirun
	python msr/bin/train_and_evaluate.py experiment=mimic/regression representation_type=agg_beat_features
	python msr/bin/train_and_evaluate.py experiment=ptbxl/regression representation_type=agg_beat_features \
		model.C=0.1 model.solver=newton-cg model.max_iter=2000 --multirun


decision_tree_experiments:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/decision_tree,sleep_edf/decision_tree,mimic/decision_tree representation_type=whole_signal_waveforms \
		model.max_depth=64 model.min_samples_split=2 model.min_samples_leaf=1 model.max_features=0.7 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/decision_tree,sleep_edf/decision_tree,mimic/decision_tree representation_type=whole_signal_features \
		model.max_depth=16 model.min_samples_split=2 model.min_samples_leaf=4 model.max_features=0.5 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/decision_tree,mimic/decision_tree representation_type=agg_beat_waveforms \
		model.max_depth=16 model.min_samples_split=2 model.min_samples_leaf=1 model.max_features=0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/decision_tree,mimic/decision_tree representation_type=agg_beat_features \
		model.max_depth=16 model.min_samples_split=8 model.min_samples_leaf=1 model.max_features=0.7 --multirun


lgbm_experiments:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/lgbm,sleep_edf/lgbm,mimic/lgbm representation_type=whole_signal_waveforms \
		model.colsample_bytree=0.1 model.max_depth=4 model.num_leaves=31 model.n_estimators=600 model.learning_rate=0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/lgbm,sleep_edf/lgbm,mimic/lgbm representation_type=whole_signal_features \
		model.colsample_bytree=0.1 model.max_depth=64 model.num_leaves=31 model.n_estimators=600 model.learning_rate=0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/lgbm,mimic/lgbm representation_type=agg_beat_waveforms \
		model.colsample_bytree=0.9 model.max_depth=32 model.num_leaves=31 model.n_estimators=600 model.learning_rate=0.1 --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/lgbm,mimic/lgbm representation_type=agg_beat_features \
		model.colsample_bytree=0.9 model.max_depth=32 model.num_leaves=31 model.n_estimators=600 model.learning_rate=0.1 --multirun


mlp_experiments:
	python msr/bin/train_and_evaluate.py experiment=ptbxl/mlp,sleep_edf/mlp,mimic/mlp representation_type=whole_signal_waveforms \
		TODO --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/mlp,sleep_edf/mlp,mimic/mlp representation_type=whole_signal_features \
		TODO --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/mlp,mimic/mlp representation_type=agg_beat_waveforms \
		TODO --multirun
	python msr/bin/train_and_evaluate.py experiment=ptbxl/mlp,mimic/mlp representation_type=agg_beat_features \
		TODO --multirun
