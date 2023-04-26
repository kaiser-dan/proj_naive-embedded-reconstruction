.PHONY: all setup clean
.DEFAULT_GOAL := all

REQUIRMENTS = environment.yaml
ENV_NAME = EmbeddedNaive

DIR_DATA = data/input/raw



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


setup: $(REQUIREMENTS)
	@echo "Creating conda environment from $(REQUIREMENTS)..."
	conda env create -f $(REQUIREMENTS)
	conda activate $(ENV_NAME)


clean:
	@echo "\n\nRemoving generated temporary files...\n\n"
	@echo "THING HAPPENS\n\n"
	@echo "Removing cached files..."
	@find ./ -name "__pycache__" -exec rm -rf {} \;
