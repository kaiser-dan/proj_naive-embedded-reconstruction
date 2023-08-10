# ========== PREFACE ==========
# Make specifications
.PHONY: all install check clean deepclean
.DEFAULT_GOAL := all

# Requirements for setup rule
REQUIRMENTS=environment.yaml
ENV_NAME=EmbeddedNaive

# Flags
INSTALL_TEST=1
INSTALL_REPRODUCE=1

# ========== Deployment ==========
all: install check clean

install:
	pip install .
	[ "${INSTALL_TEST}" = "1" ] && pip install .[test]
	[ "${INSTALL_REPRODUCE}" = "1" ] && pip install .[reproduce]

check:
	pytest tests/

# ========== Workflow reproduction ==========
# --- Data acquisition & preparations ---
get_data_tarballs:
	@echo "Retrieving 'arXiv collaboration multiplex'..."
	@if ! [ -f $(DIR_DATA)/arxiv.zip ] ;\
	then \
		curl -o $(DIR_DATA)/arxiv.zip https://manliodedomenico.com/data/arXiv-Netscience_Multiplex_Coauthorship.zip |> /dev/null;\
	else \
		echo "'arxiv collaboration multiplex' already present!" ;\
	fi
	@echo "\n\nRetrieving 'celegans connectome'..."
	@if ! [ -f $(DIR_DATA)/celegans.zip ] ;\
	then \
		curl -o $(DIR_DATA)/celegans.zip https://manliodedomenico.com/data/CElegans_Multiplex_Neuronal.zip |> /dev/null;\
	else \
		echo "'celegans connectome' already present!" ;\
	fi
	@echo "\n\nRetrieving 'drosophila genetic interaction multiplex'..."
	@if ! [ -f $(DIR_DATA)/drosophila.zip ] ;\
	then \
		curl -o $(DIR_DATA)/drosophila.zip https://manliodedomenico.com/data/Drosophila_Multiplex_Genetic.zip |> /dev/null;\
	else \
		echo "'drosophila genetic interaction multiplex' already present!" ;\
	fi
	@echo "\n\nRetrieving 'london transportation multiplex'..."
	@if ! [ -f $(DIR_DATA)/london.zip ] ;\
	then \
		curl -o $(DIR_DATA)/london.zip https://manliodedomenico.com/data/London_Multiplex_Transport.zip |> /dev/null;\
	else \
		echo "'london transportation multiplex' already present!" ;\
	fi

# TODO: Test unpacking make rule
unpack_data: $(DIR_DATA)/arxiv.zip $(DIR_DATA)/celegans.zip $(DIR_DATA)/drosophila.zip $(DIR_DATA)/london.zip
	@echo "Unpacking 'arXiv collaboration multiplex'..."
	gunzip $(DIR_DATA)/arxiv.zip
	cp $(DIR_DATA)/PATH/TO/EDGELIST.edges $(DIR_DATA_PREPROCESSED)/multiplex_system-arxiv.edgelist
	@echo "''arXiv collaboration multiplex' unpacked!"
	@echo "Unpacking 'celegans connectome'..."
	gunzip $(DIR_DATA)/celegans.zip
	cp $(DIR_DATA)/PATH/TO/EDGELIST.edges $(DIR_DATA_PREPROCESSED)/multiplex_system-celegans.edgelist
	@echo "'celegans connectome' unpacked!"
	@echo "Unpacking 'drosophila genetic interaction multiplex'..."
	gunzip $(DIR_DATA)/drosophila.zip
	cp $(DIR_DATA)/PATH/TO/EDGELIST.edges $(DIR_DATA_PREPROCESSED)/multiplex_system-drosophila.edgelist
	@echo "'drosophila genetic interaction multiplex' unpacked!"
	@echo "Unpacking 'london transportation multiplex'..."
	gunzip $(DIR_DATA)/london.zip
	cp $(DIR_DATA)/PATH/TO/EDGELIST.edges $(DIR_DATA_PREPROCESSED)/multiplex_system-london.edgelist
	@echo "'london transportation multiplex' unpacked!"

# ========== Repo management ==========
# --- Cleaning ---
clean: clean_tmp clean_logs
deepclean: clean clean_caches clean_downloaded clean_build

clean_tmp:
	@echo "Removing generated temporary files"
	@find ./ -type f -name "*.tmp" -delete

clean_logs:
	@echo "Removing generated log files"
	@find ./ -type f -name "*.log" -delete

clean_caches:
	@echo "Removing python cache files"
	@find ./ -type f -regextype egrep -regex ".*\.py[cod]" -delete
	@find ./ -type d -name "__pycache__" -exec rm -rf {} \;

clean_downloaded:
	@echo "Removing downloaded multiplex data...\n"
	@find ./ -regextype egrep -regex ".*(arxiv|celegans|drosophila|london).*" -delete

clean_build:
	@echo "Removing build files"
	@rm -rf build/
	@rm -rf src/embmplxrec.egg-info
