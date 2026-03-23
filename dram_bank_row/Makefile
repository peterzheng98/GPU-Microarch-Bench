# dram_bank_row – DRAM bank & row size discovery
#
# Build:
#   make                    # fat binary (sm_80 + sm_90) for multi-GPU
#   make ARCH=ampere        # Ampere only  (sm_80)
#   make ARCH=hopper        # Hopper only  (sm_90)
#   make ARCH=both          # explicit fat binary

NVCC       ?= nvcc
ARCH       ?= both
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
