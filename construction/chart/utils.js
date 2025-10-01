// Sadly we can't specify file's save direction for the blowser's fault
// const CHORD_CHART_SAVE_DIR_PNG = "/png/chord";
// const CHORD_CHART_SAVE_DIR_SVG = "/svg/chord";
// const SANKEY_CHART_SAVE_DIR_PNG = "/png/sankey";
// const SANKEY_CHART_SAVE_DIR_SVG = "/svg/sankey";
// const FUNNEL_CHART_SAVE_DIR_PNG = "/png/funnel";
// const FUNNEL_CHART_SAVE_DIR_SVG = "/svg/funnel"; 

// change the chart index to generate different chart figures for batch export
const CURRENT_CHART_INDEX = 2;
const chartTypes = [
    { rootDir: CHORD_CSV_ROOT_DIR,  processFunction: createChordChart,  type : "chord"  },
    { rootDir: SANKEY_CSV_ROOT_DIR, processFunction: createSankeyChart, type : "sankey" },
    { rootDir: FUNNEL_CSV_ROOT_DIR, processFunction: createFunnelChart, type : "funnel" },
];

// use {chartType}_theme_i as file name
const THEME_LIST = [
    "transportation_and_logistics",
    "tourism_and_hospitality",
    "business_and_finance",
    "real_estate_and_housing_market",
    "healthcare_and_health",
    "retail_and_ecommerce",
    "human_resources_and_employee_management",
    "sports_and_entertainment",
    "education_and_academics",
    "food_and_beverage_industry",
    "science_and_engineering",
    "agriculture_and_food_production",
    "energy_and_utilities",
    "cultural_trends_and_influences",
    "social_media_and_digital_media_and_streaming"
];

document.getElementById('processBtn').addEventListener('click', async () => {
    await processCSVFiles();
});

// we can slice the task into smaller pieces to avoid the browser's memory limit
// each generation will process 100 files
const START_INDEX = 1;
const GERNATION_STEP_LENTH = 1;
const END_INDEX = START_INDEX + GERNATION_STEP_LENTH - 1;

async function processCSVFiles() {
    let rootDir = chartTypes[CURRENT_CHART_INDEX].rootDir;
    let processFunction = chartTypes[CURRENT_CHART_INDEX].processFunction;
    let chartType = chartTypes[CURRENT_CHART_INDEX].type;

    for (const theme of THEME_LIST) {
        const file_prefix = `${chartType}_${theme}`;
        for (let i = START_INDEX; i <= END_INDEX; i++) {
            const file = `${rootDir}/${file_prefix}_${i}.csv`;

            try {
                await processFunction(file);
            } catch (error) {
                console.error(`处理文件 ${file} 失败:`, error);
            }
        }
    }
}

function addWhiteBackgroundToSVG(svgElement) {
    if (!svgElement) return;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", "0");
    rect.setAttribute("y", "0");

    rect.setAttribute("width", svgElement.getAttribute("width") || svgElement.clientWidth);
    rect.setAttribute("height", svgElement.getAttribute("height") || svgElement.clientHeight);

    rect.setAttribute("fill", "#ffffff");
    svgElement.insertBefore(rect, svgElement.firstChild);
}


function downloadChart(chartOrElementId, svgFilename, pngFilename, isChart = false) {
    let svgElement, svgString;

    if (isChart) {
        // Assuming `chart` is an instance of a visualization library (e.g., ECharts)
        const chart = chartOrElementId;
        svgElement = chart.getDom().querySelector('svg');
        const serializer = new XMLSerializer();
        svgString = serializer.serializeToString(svgElement);

        // Download SVG
        const svgDataUrl = chart.getDataURL({
            type: 'svg',
            backgroundColor: '#ffffff'
        });
        const svgLink = document.createElement('a');
        svgLink.href = svgDataUrl;
        svgLink.download = svgFilename;
        document.body.appendChild(svgLink);
        svgLink.click();
        document.body.removeChild(svgLink);
    } else {
        // Assuming `chartOrElementId` is the ID of an SVG element
        svgElement = document.getElementById(chartOrElementId);
        addWhiteBackgroundToSVG(svgElement);
        const serializer = new XMLSerializer();
        svgString = serializer.serializeToString(svgElement);

        // Download SVG
        const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
        const svgUrl = URL.createObjectURL(svgBlob);
        const svgLink = document.createElement("a");
        svgLink.href = svgUrl;
        svgLink.download = svgFilename;
        document.body.appendChild(svgLink);
        svgLink.click();
        document.body.removeChild(svgLink);
    }

    // Download PNG
    const img = new Image();
    const canvas = document.createElement("canvas");
    canvas.width = svgElement.width.baseVal.value;
    canvas.height = svgElement.height.baseVal.value;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FFFFFF"; // Set background color to white
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const url = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));

    img.onload = function () {
        ctx.drawImage(img, 0, 0);
        canvas.toBlob(function (blob) {
            const pngUrl = URL.createObjectURL(blob);
            const pngLink = document.createElement("a");
            pngLink.href = pngUrl;
            pngLink.download = pngFilename;
            document.body.appendChild(pngLink);
            pngLink.click();
            document.body.removeChild(pngLink);
        }, "image/png");
    };
    img.src = url;
}

// createChordChart("csv/chord/chord_energy_and_utilities_1.csv", debugMode = DEBUG_MODE);
// createFunnelChart("csv/funnel/funnel_agriculture_and_food_production_1.csv", debugMode = DEBUG_MODE);
// createSankeyChart('csv/sankey/sankey_social_media_and_digital_media_and_streaming_1.csv', debugMode = DEBUG_MODE);
// createSankeyChart('csv/sankey/sankey_business_and_finance_1.csv', debugMode = DEBUG_MODE);
// createSankeyChart('csv/sankey/sankey_social_media_and_digital_media_and_streaming_1.csv', debugMode = DEBUG_MODE);