ROOT := $(CURDIR)
SUBMODULE_DIRS := $(shell git config --file .gitmodules --get-regexp path | awk '{print $$2}')
QUARTO_PROJECTS := $(foreach dir,$(SUBMODULE_DIRS),$(dir)/src)
GIT ?= git
QUARTO_VERSION ?= 1.8.26
QUARTO_HOME := $(HOME)/.local/quarto
QUARTO_TARBALL := https://quarto.org/download/latest/quarto-linux-amd64.tar.gz
Q_BIN := $(shell command -v quarto 2>/dev/null || echo "$(QUARTO_HOME)/bin/quarto")

.PHONY: init-submodules pull-submodules push-submodules update-submodules reset-submodules status install-quarto update-quarto verify-quarto preview publish

install-quarto:
	@if [ ! -x "$(QUARTO_HOME)/bin/quarto" ]; then \
	    echo "Installing Quarto $(QUARTO_VERSION)..."; \
	    wget -O /tmp/quarto.tgz $(QUARTO_TARBALL); \
	    mkdir -p $(QUARTO_HOME); \
	    tar -xzf /tmp/quarto.tgz -C $(QUARTO_HOME) --strip-components=1; \
	    rm /tmp/quarto.tgz; \
	    echo 'export PATH="$(QUARTO_HOME)/bin:$$PATH"' >> $(HOME)/.bashrc; \
	    echo "Run 'source ~/.bashrc' or restart terminal to load PATH."; \
	else \
	    echo "Quarto already installed."; \
	fi

update-quarto:
	@echo "Updating Quarto..."
	@rm -rf $(QUARTO_HOME)
	@mkdir -p $(QUARTO_HOME)
	@wget -O /tmp/quarto.tgz $(QUARTO_TARBALL)
	@tar -xzf /tmp/quarto.tgz -C $(QUARTO_HOME) --strip-components=1
	@rm /tmp/quarto.tgz
	@echo "Quarto updated."

verify-quarto:
	@echo "Using Quarto binary: $(Q_BIN)"
	@if [ ! -x "$(Q_BIN)" ]; then \
		echo "Quarto not installed or binary missing"; exit 1; \
	fi
	@$(Q_BIN) --version

preview:
	@for project in $(QUARTO_PROJECTS); do \
		echo "=== Previewing $$project ==="; \
		cd $$project && $(Q_BIN) preview . --no-browser || exit 1; \
	done

publish:
	@for project in $(QUARTO_PROJECTS); do \
		echo "=== Publishing $$project ==="; \
		cd $$project && $(Q_BIN) publish huggingface || exit 1; \
	done

init-submodules:
	$(GIT) submodule update --init --recursive

pull-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Pulling submodule $$dir"; \
		cd $$dir && $(GIT) pull origin main || true; \
	done

update-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Committing and pushing $$dir"; \
		cd $$dir && \
		$(GIT) add . && \
		$(GIT) commit -m "Update submodule $$dir" || true; \
		$(GIT) push origin main || true; \
	done

push-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Pushing $$dir"; \
		cd $$dir && $(GIT) push origin main || true; \
	done

reset-submodules:
	@for dir in $(SUBMODULE_DIRS); do \
		echo ">>> Resetting $$dir"; \
		cd $$dir && $(GIT) fetch origin && $(GIT) reset --hard origin/main; \
	done

status:
	@echo "==== MAIN REPO ===="
	$(GIT) status
	@echo "==== SUBMODULES ===="
	@for dir in $(SUBMODULE_DIRS); do \
		echo "--- $$dir ---"; \
		cd $$dir && $(GIT) status; \
	done
