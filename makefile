.PHONY: all setup clean
.DEFAULT_GOAL := all

REQUIRMENTS=environment.yaml
ENV_NAME=EmbeddedNaive

DIR_DATA=data/input/raw
DIR_DATA_PREPROCESSED=data/input/preprocessed/edgelists

all: get_data_tarballs unpack_data clean


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


setup: $(REQUIREMENTS)
	@echo "Creating conda environment from $(REQUIREMENTS)..."
	conda env create -f $(REQUIREMENTS)
	conda activate $(ENV_NAME)


clean:
	@echo "\n\nRemoving generated temporary files...\n\n"
	@echo "THING HAPPENS\n\n"
	@echo "\n\nRemoving downloaded multiplex data...\n\n"
	@find $(DATA_DIR) -regextype posix-extended -regex ".*(arxiv|celegans|drosophila|london).*" -delete
	@echo "Removing pycache files..."
	@find ./ -name "__pycache__" -exec rm -rf {} \;
