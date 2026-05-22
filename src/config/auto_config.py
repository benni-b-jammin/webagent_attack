# ============================================================
# General
# ============================================================

DRY_RUN = False
START_STEP = 1
END_STEP = 6

# ============================================================
# Step 1: Website retrieval / capture
# ============================================================

QUERY_TYPES_FILE = "data/page_types.txt"
N_WEBSITES = 50
N_SEARCH_QUERIES = 12
URLS_PER_QUERY = 10
CAPTURED_SITES_DIR = "src/data/datasets/auto_data"

# ============================================================
# Step 2: Memory audit
# ============================================================

AUDIT_OUTDIR = "src/results/website_memory_audit"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TRIGGER_CONFIG_DIR = "src/config/narrow_triggers"
DEFAULT_TRIGGER_NAME = "trigger_default.yaml"
MEASURE_TRIGGER = False
TRIGGER_AUDIT_STEPS = 5
DELETE_FAILED_WEBNAV = False
DELETE_FAILED_TRIGGER = False

# ============================================================
# Step 3: Generate actions / prompts / trigger YAMLs
# ============================================================

PROMPTS_DIR = "src/data/test_prompts"
META_DIR = "src/data/task_meta"
ACTIONS_SUMMARY_CSV = "results/generated_site_actions_summary.csv"
ACTION_MODEL = "gpt-4o-mini"
N_REGULAR_ACTIONS = 10
SKIP_EXISTING_ACTIONS = False

# ============================================================
# Step 4: Generate triggers
# ============================================================

N_TRIGGERS_PER_SITE = 1
TRIGGER_ALGO = "gcg"
SKIP_DEFAULT_TRIGGER_CONFIG = True

# ============================================================
# Step 5: Run trigger tests
# ============================================================

TRIGGER_DIR = "src/data/triggers"
AUTO_CONFIG = "src/config/auto_runs/auto_default.yaml"
HEADLESS_TESTS = "true"
LATEST_TRIGGER_ONLY = False
LIMIT_TEST_SITES = None
RESULTS_DIR = "results/pipeline_runs"

# ============================================================
# Step 6: Summaries
# ============================================================

SUMMARY_OUTDIR = "results/final_summary"
ATTACK_SUMMARY_CSV = None
MEMORY_SUMMARY_CSV = "src/results/website_memory_audit/website_summary_table.csv"
MEMORY_FEATURE_CSV = "src/results/website_memory_audit/website_feature_runtime_table.csv"