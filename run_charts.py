from src.analysis.charts import ThesisChartGenerator
import os
import logging
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.config.settings import Settings
    from src.analysis.charts import ThesisChartGenerator
except ImportError as e:
    logger.error(f"FATAL: Could not import necessary modules: {e}")
    logger.error(
        "Ensure Settings is in src.config and ThesisChartGenerator is in src.plotting")
    exit()
except Exception as e:
    logger.error(f"An unexpected error occurred during imports: {e}")
    exit()


def run_chart_creation():
    logger.info("--- Starting Chart Generation Process ---")

    try:
        settings = Settings()
        logger.info("Settings loaded.")
        if not os.path.isdir(settings.output_base_dir):
            logger.warning(
                f"Base output directory specified in settings does not exist: {settings.output_base_dir}. Plot saving may fail if subdirs don't exist.")

    except Exception as e:
        logger.error(f"FATAL: Failed to load settings: {e}")
        return

    try:
        chart_gen = ThesisChartGenerator(settings)
    except Exception as e:
        logger.error(
            f"FATAL: Failed to initialize ThesisChartGenerator: {e}", exc_info=True)
        logger.error(
            "Ensure all required input CSV files exist in the correct directories.")
        return

    try:
        chart_gen.generate_all_plots()
    except Exception as e:
        logger.error(
            f"FATAL: An error occurred during plot generation: {e}", exc_info=True)
        return

    logger.info("--- Chart Generation Process Finished Successfully ---")


if __name__ == "__main__":
    run_chart_creation()
