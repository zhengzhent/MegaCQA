// Chart settings
const CHART_WIDTH = 1920;
const CHART_HEIGHT = 1440;

const NUM_OF_THEMES = 15;
const CSV_FILE_NUM_BY_THEME = 1000;
const CSV_FILE_NUM_TOTAL = NUM_OF_THEMES * CSV_FILE_NUM_BY_THEME;

const CHORD_CSV_ROOT_DIR = "/csv/chord";
const SANKEY_CSV_ROOT_DIR = "/csv/sankey";
const FUNNEL_CSV_ROOT_DIR = "/csv/funnel";


// After so many trails and errors, i found out the safe minimum time gap is 200ms
const RELOAD_TIME_GAP = 200;

// set to false if really need to generate
const DEBUG_MODE = true; 
