#!/bin/bash
# ============================================================================
# RALFS Configuration Setup Script
# ============================================================================
set -e

echo "🔧 Setting up RALFS configuration system..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Create directory structure
echo "Creating config directory structure..."
mkdir -p configs/{data,retriever,generator,train,experiment}

echo "✅ Configuration directories created"

# Validate required configs
echo "Validating configs..."
if [ ! -f "configs/ralfs.yaml" ]; then
    echo "⚠️  configs/ralfs.yaml not found - please create it"
    exit 1
fi

echo "✅ Configuration setup complete!"
echo "Config directories:"
ls -la configs/
# Step 1: Create directory structure
# ============================================================================
echo -e "\n${YELLOW}Step 1: Creating config directory structure...${NC}"

mkdir -p configs/data
mkdir -p configs/retriever
mkdir -p configs/generator
mkdir -p configs/train
mkdir -p configs/experiment

echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================================
# Step 2: Validate existing config files
# ============================================================================
echo -e "\n${YELLOW}Step 2: Validating existing configs...${NC}"

if [ -f "configs/ralfs.yaml" ]; then
    echo -e "${GREEN}✓ configs/ralfs.yaml exists${NC}"
else
    echo -e "${RED}✗ configs/ralfs.yaml missing${NC}"
    exit 1
fi

# Data configs
for config in default arxiv govreport debug; do
    if [ -f "configs/data/${config}.yaml" ]; then
        echo -e "${GREEN}✓ configs/data/${config}.yaml exists${NC}"
    else
        echo -e "${YELLOW}⚠ configs/data/${config}.yaml missing (optional)${NC}"
    fi
done

# Retriever configs
for config in hybrid dense sparse; do
    if [ -f "configs/retriever/${config}.yaml" ]; then
        echo -e "${GREEN}✓ configs/retriever/${config}.yaml exists${NC}"
    else
        echo -e "${RED}✗ configs/retriever/${config}.yaml missing${NC}"
        exit 1
    fi
done

# Generator configs
for config in fid fid_base fid_xl; do
    if [ -f "configs/generator/${config}.yaml" ]; then
        echo -e "${GREEN}✓ configs/generator/${config}.yaml exists${NC}"
    else
        echo -e "${YELLOW}⚠ configs/generator/${config}.yaml missing (optional)${NC}"
    fi
done

# Train configs
for config in default a100 debug multi_gpu; do
    if [ -f "configs/train/${config}.yaml" ]; then
        echo -e "${GREEN}✓ configs/train/${config}.yaml exists${NC}"
    else
        echo -e "${YELLOW}⚠ configs/train/${config}.yaml missing (optional)${NC}"
    fi
done

# ============================================================================
# Step 3: Test config loading with Python
# ============================================================================
echo -e "\n${YELLOW}Step 3: Testing config loading...${NC}"

python3 << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
from omegaconf import OmegaConf

try:
    # Test main config
    cfg = OmegaConf.load("configs/ralfs.yaml")
    print(f"✓ Main config loaded successfully")
    print(f"  - Task: {cfg.get('task', 'N/A')}")
    
    # Test data config
    data_cfg = OmegaConf.load("configs/data/default.yaml")
    print(f"✓ Data config loaded successfully")
    print(f"  - Dataset: {data_cfg.get('dataset', 'N/A')}")
    
    # Test retriever config
    retriever_cfg = OmegaConf.load("configs/retriever/hybrid.yaml")
    print(f"✓ Retriever config loaded successfully")
    print(f"  - Type: {retriever_cfg.get('type', 'N/A')}")
    
    # Test generator config
    gen_cfg = OmegaConf.load("configs/generator/fid.yaml")
    print(f"✓ Generator config loaded successfully")
    print(f"  - Model: {gen_cfg.model.get('name', 'N/A')}")
    
    # Test train config
    train_cfg = OmegaConf.load("configs/train/default.yaml")
    print(f"✓ Train config loaded successfully")
    print(f"  - Batch size: {train_cfg.training.get('batch_size', 'N/A')}")
    
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error loading configs: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All configs loaded successfully${NC}"
else
    echo -e "${RED}✗ Config loading failed${NC}"
    exit 1
fi

# ============================================================================
# Step 4: Test config integration with RALFS
# ============================================================================
echo -e "\n${YELLOW}Step 4: Testing RALFS config integration...${NC}"

python3 << 'PYTHON_SCRIPT'
import sys
try:
    from ralfs.core import load_config
    
    # Test default config
    config = load_config("configs/ralfs.yaml")
    print(f"✓ RALFSConfig loaded successfully")
    print(f"  - Task: {config.task}")
    print(f"  - Dataset: {config.data.dataset}")
    print(f"  - Retriever: {config.retriever.type}")
    print(f"  - Generator: {config.generator.model_name}")
    
    # Test with overrides
    config = load_config(
        "configs/ralfs.yaml",
        overrides=["data.dataset=pubmed", "train.batch_size=4"]
    )
    print(f"✓ Config overrides work")
    print(f"  - Dataset override: {config.data.dataset}")
    print(f"  - Batch size override: {config.train.batch_size}")
    
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error loading RALFS config: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ RALFS config integration works${NC}"
else
    echo -e "${RED}✗ RALFS config integration failed${NC}"
    exit 1
fi

# ============================================================================
# Step 5: Create .env.example for API keys
# ============================================================================
echo -e "\n${YELLOW}Step 5: Creating .env.example...${NC}"

cat > .env.example << 'EOF'
# RALFS Environment Variables

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=ralfs-experiments
WANDB_ENTITY=your_username_or_team

# Hugging Face (optional, for private models)
HF_TOKEN=your_huggingface_token_here

# GPU Settings
CUDA_VISIBLE_DEVICES=0

# Hydra Settings
HYDRA_FULL_ERROR=1
EOF

echo -e "${GREEN}✓ .env.example created${NC}"

# ============================================================================
# Step 6: Print summary
# ============================================================================
echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Configuration setup complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Quick Start:${NC}"
echo -e "  1. Copy .env.example to .env and fill in your API keys"
echo -e "  2. Test config loading:"
echo -e "     ${BLUE}ralfs train --cfg job${NC}"
echo -e "  3. Run a debug training:"
echo -e "     ${BLUE}ralfs train train=debug data=debug${NC}"
echo -e "  4. View all available configs:"
echo -e "     ${BLUE}ls -R configs/${NC}"

echo -e "\n${YELLOW}Example Commands:${NC}"
echo -e "  # Default training (T4 GPU)"
echo -e "  ${BLUE}ralfs train${NC}"
echo -e ""
echo -e "  # A100 GPU training"
echo -e "  ${BLUE}ralfs train train=a100 generator=fid_xl${NC}"
echo -e ""
echo -e "  # Ablation: Dense-only retrieval"
echo -e "  ${BLUE}ralfs train retriever=dense${NC}"
echo -e ""
echo -e "  # Custom overrides"
echo -e "  ${BLUE}ralfs train data.dataset=pubmed train.batch_size=4${NC}"

echo -e "\n${GREEN}Setup complete! 🚀${NC}\n"