#! /bin/bash

# Set default values
INTELLIGENCE_URL=${AGENT_URL:-"http://0.0.0.0:10004"}
PROBLEMS_FILE=${PROBLEMS_FILE:-"./data/example_problem.jsonl"}
K_VALUES=${K_VALUES:-"1 3 5"}
N_COMPLETIONS=${N_COMPLETIONS:-"1"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results"}
BASE_URL=${BASE_URL:-}
MODEL=${MODEL:-"gpt-4o"}

# Parse command-line arguments to override defaults if provided
for arg in "$@"; do
  case $arg in
    --intelligence_url=*)
      INTELLIGENCE_URL="${arg#*=}"
      shift
      ;;
    --problems_file=*)
      PROBLEMS_FILE="${arg#*=}"
      shift
      ;;
    --k=*)
      K_VALUES="${arg#*=}"
      shift
      ;;
    --n_completions=*)
      N_COMPLETIONS="${arg#*=}"
      shift
      ;;
    --output_dir=*)
      OUTPUT_DIR="${arg#*=}"
      shift
      ;;
    --base_url=*)
      BASE_URL="${arg#*=}"
      shift
      ;;
    --model=*)
      MODEL="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      ;;
  esac
done

# Display the final configuration for verification
# echo "Using the following configuration:"
# echo "INTELLIGENCE_URL: $INTELLIGENCE_URL"
# echo "PROBLEMS_FILE:    $PROBLEMS_FILE"
# echo "K_VALUES:         $K_VALUES"
# echo "N_COMPLETIONS:    $N_COMPLETIONS"
# echo "OUTPUT_DIR:       $OUTPUT_DIR"
# echo "BASE_URL:         $BASE_URL"
# echo "MODEL:            $MODEL"


# Run evaluation
python evaluate_from_api.py \
  --intelligence_url "$INTELLIGENCE_URL" \
  --problems_file "$PROBLEMS_FILE" \
  --k $K_VALUES \
  --n_completions "$N_COMPLETIONS" \
  --output_dir "$OUTPUT_DIR" \
  --base_url "$BASE_URL" \
  --model "$MODEL"

# tail -f /dev/null
