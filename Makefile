# GPU-Microarch-Bench – top-level build
#
#   make                 Build all benchmarks (Ampere by default)
#   make ARCH=hopper     Build for Hopper
#   make ARCH=both       Fat binary for Ampere + Hopper
#   make clean           Remove all build artefacts

ARCH ?= both

SUBDIRS := dram_bank_row

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ ARCH=$(ARCH)

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
