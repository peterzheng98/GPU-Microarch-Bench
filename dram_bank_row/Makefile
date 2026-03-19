# dram_bank_row – DRAM bank & row size discovery
#
# Build:
#   make                    # Ampere (sm_80) by default
#   make ARCH=hopper        # Hopper (sm_90)
#   make ARCH=both          # fat binary with both architectures

NVCC       ?= nvcc
ARCH       ?= ampere
NVCC_FLAGS := -O3 -std=c++17 -lineinfo

ifeq ($(ARCH),ampere)
  NVCC_FLAGS += -gencode arch=compute_80,code=sm_80
else ifeq ($(ARCH),hopper)
  NVCC_FLAGS += -gencode arch=compute_90,code=sm_90
else ifeq ($(ARCH),both)
  NVCC_FLAGS += -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_90,code=sm_90
else
  $(error Unknown ARCH=$(ARCH). Use ampere, hopper, or both.)
endif

TARGET := dram_bank_row
SRC    := dram_bank_row.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	$(RM) $(TARGET)
