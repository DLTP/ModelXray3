class ArithmeticNN {
    constructor() {
        this.initWeights();
        this.totalSteps = 0;
    }

    initWeights() {
        // Upgrade: 6 -> 64 -> 32 -> 1. This is a very dense "high-resolution" brain.
        this.w1 = this.randomMatrix(6, 64, Math.sqrt(2/6));
        this.b1 = this.randomMatrix(1, 64, 0);
        this.w2 = this.randomMatrix(64, 32, Math.sqrt(2/64));
        this.b2 = this.randomMatrix(1, 32, 0);
        this.w3 = this.randomMatrix(32, 1, Math.sqrt(2/32));
        this.b3 = this.randomMatrix(1, 1, 0);
        this.learningRate = 0.05; // Slightly faster learning
    }

    randomMatrix(r, c, scale) {
        return Array.from({ length: r }, () => 
            Array.from({ length: c }, () => (Math.random() * 2 - 1) * scale)
        );
    }

    relu(x) { return Math.max(0, x); }
    drelu(x) { return x > 0 ? 1 : 0; }

    forward(input) {
        this.input = [input];
        this.z1 = this.matmul(this.input, this.w1);
        this.h1 = this.addBias(this.z1, this.b1).map(row => row.map(v => this.relu(v)));
        this.z2 = this.matmul(this.h1, this.w2);
        this.h2 = this.addBias(this.z2, this.b2).map(row => row.map(v => this.relu(v)));
        this.z3 = this.matmul(this.h2, this.w3);
        this.output = this.addBias(this.z3, this.b3);
        return this.output[0][0];
    }

    train(input, target) {
        const pred = this.forward(input);
        
        // Critical Fix: We scale the error to make "Small" additions matter as much as "Large" multiplications.
        const error = (pred - target);

        let d_output = error; 
        let d_w3 = this.matmul(this.transpose(this.h2), [[d_output]]);
        let d_b3 = [[d_output]];

        let d_h2 = this.matmul([[d_output]], this.transpose(this.w3));
        let d_z2 = d_h2.map((row, i) => row.map((val, j) => val * this.drelu(this.h2[i][j])));
        
        let d_w2 = this.matmul(this.transpose(this.h1), d_z2);
        let d_b2 = d_z2;

        let d_h1 = this.matmul(d_z2, this.transpose(this.w2));
        let d_z1 = d_h1.map((row, i) => row.map((val, j) => val * this.drelu(this.h1[i][j])));

        let d_w1 = this.matmul(this.transpose(this.input), d_z1);
        let d_b1 = d_z1;

        this.w3 = this.update(this.w3, d_w3);
        this.b3 = this.update(this.b3, d_b3);
        this.w2 = this.update(this.w2, d_w2);
        this.b2 = this.update(this.b2, d_b2);
        this.w1 = this.update(this.w1, d_w1);
        this.b1 = this.update(this.b1, d_b1);

        return Math.pow(error, 2); 
    }

    matmul(a, b) {
        let result = Array.from({ length: a.length }, () => Array(b[0].length).fill(0));
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
                for (let k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    addBias(a, b) {
        return a.map((row, i) => row.map((val, j) => val + b[0][j]));
    }

    transpose(a) {
        return a[0].map((_, i) => a.map(row => row[i]));
    }

    update(matrix, gradient) {
        return matrix.map((row, i) => row.map((val, j) => val - this.learningRate * gradient[i][j]));
    }
}

const model = new ArithmeticNN();
const inputNorm = 20;   
const outputNorm = 400; 
let activeDataset = []; // Decoupled storage for training data

function encodeOp(op) {
    if (op === '+') return [1, 0, 0, 0];
    if (op === '-') return [0, 1, 0, 0];
    if (op === '*') return [0, 0, 1, 0];
    if (op === '/') return [0, 0, 0, 1];
    return [0,0,0,0];
}

async function logText(containerId, text, color = 'inherit') {
    const container = document.getElementById(containerId);
    const line = document.createElement('div');
    line.style.color = color;
    line.style.marginBottom = '5px';
    line.innerText = '> ' + text;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
    await new Promise(r => setTimeout(r, 600)); 
}

// STEP 1: GENERATE DATASET
function generateDataset(size) {
    activeDataset = [];
    for (let i = 0; i < size; i++) {
        const ops = ['+', '-', '*', '/'];
        const op = ops[Math.floor(Math.random() * ops.length)];
        const n1 = Math.floor(Math.random() * inputNorm);
        const n2 = Math.floor(Math.random() * (inputNorm - 1)) + 1;
        let res = 0;
        if (op === '+') res = n1 + n2;
        if (op === '-') res = n1 - n2;
        if (op === '*') res = n1 * n2;
        if (op === '/') res = n1 / n2;

        activeDataset.push({
            input: [n1 / inputNorm, n2 / inputNorm, ...encodeOp(op)],
            target: res / outputNorm,
            raw: { n1, n2, op, res }
        });
    }
    document.getElementById('dataset-status').innerText = `Dataset Created: ${activeDataset.length} rows ready in memory.`;
}

// SAVE DATASET TO CSV
function saveDatasetToCSV() {
    if (activeDataset.length === 0) {
        alert("No data to save!");
        return;
    }
    const headers = ["n1", "n2", "op", "res"];
    const rows = activeDataset.map(d => [d.raw.n1, d.raw.n2, d.raw.op, d.raw.res]);
    let csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\n" + rows.map(e => e.join(",")).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "calculator_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// LOAD DATASET FROM CSV
function loadDatasetFromCSV(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const lines = text.split('\n');
        activeDataset = [];
        
        // Skip header
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(',');
            if (cols.length < 4) continue;
            
            const n1 = parseFloat(cols[0]);
            const n2 = parseFloat(cols[1]);
            const op = cols[2].trim();
            const res = parseFloat(cols[3]);

            activeDataset.push({
                input: [n1 / inputNorm, n2 / inputNorm, ...encodeOp(op)],
                target: res / outputNorm,
                raw: { n1, n2, op, res }
            });
        }
        document.getElementById('dataset-status').innerText = `Dataset Loaded: ${activeDataset.length} rows.`;
    };
    reader.readAsText(file);
}

// SAVE MODEL TO FILE (JSON)
function saveModelToFile() {
    const modelData = {
        w1: model.w1, b1: model.b1,
        w2: model.w2, b2: model.b2,
        w3: model.w3, b3: model.b3,
        totalSteps: model.totalSteps
    };
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(modelData));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "ai_model.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.removeChild();
}

// LOAD MODEL FROM FILE (JSON)
function loadModelFromFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            model.w1 = data.w1; model.b1 = data.b1;
            model.w2 = data.w2; model.b2 = data.b2;
            model.w3 = data.w3; model.b3 = data.b3;
            model.totalSteps = data.totalSteps || 0;
            
            updateStatus(0); // Refresh UI
            alert("Model loaded successfully!");
        } catch (err) {
            alert("Error loading model: " + err.message);
        }
    };
    reader.readAsText(file);
}

// STEP 2: TRAIN ON DATASET
async function runMassTraining() {
    const epochInput = parseInt(document.getElementById('train-epoch').value) || 1;
    const batchSize = activeDataset.length;
    
    if (batchSize === 0) {
        alert("Please generate a dataset first!");
        return;
    }

    let totalLoss = 0;
    document.getElementById('mass-train-btn').innerText = "Training...";
    document.getElementById('mass-train-btn').disabled = true;

    for (let e = 0; e < epochInput; e++) {
        totalLoss = 0;
        for (let i = 0; i < batchSize; i++) {
            const sample = activeDataset[i];
            totalLoss += model.train(sample.input, sample.target);
            model.totalSteps++;
            
            // UI Update every 500 steps to keep browser alive
            if (model.totalSteps % 500 === 0) {
                await new Promise(r => setTimeout(r, 0));
                updateStatus(totalLoss / (i + 1));
            }
        }
        console.log(`Epoch ${e+1} Complete. Avg Loss: ${totalLoss/batchSize}`);
    }

    document.getElementById('mass-train-btn').innerText = "Run Batch Training";
    document.getElementById('mass-train-btn').disabled = false;
    updateStatus(totalLoss / batchSize);
}

async function manualTrain() {
    const inputStr = document.getElementById('manual-input').value; 
    const regex = /([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)\s*=\s*([\d\.]+)/;
    const match = inputStr.match(regex);
    
    const loggerId = 'manual-training-log';
    const log = document.getElementById(loggerId);
    log.innerHTML = '';

    if (!match) {
        await logText(loggerId, "Error: Invalid format. Please use '1 + 2 = 3'", "#ff6d91");
        return;
    }

    const n1 = parseFloat(match[1]);
    const op = match[2];
    const n2 = parseFloat(match[3]);
    const targetRaw = parseFloat(match[4]);

    const sample = {
        input: [n1 / inputNorm, n2 / inputNorm, ...encodeOp(op)],
        target: targetRaw / outputNorm,
        raw: { n1, n2, op, res: targetRaw }
    };

    await logText(loggerId, `Concept: Starting training for specifically provided row.`);
    await logText(loggerId, `Step 1: AI converts ${n1} and ${n2} into matrix values.`);
    
    const predBefore = model.forward(sample.input) * outputNorm;
    await logText(loggerId, `Current Prediction is: ${predBefore.toFixed(2)}. Targets is: ${targetRaw}.`);
    
    const loss = model.train(sample.input, sample.target);
    await logText(loggerId, `Step 2: Calculating Gradient. We calculate exactly how much every number in W1/W2/W3 needs to tilt.`);
    await logText(loggerId, `Step 3: Backpropagation! Nudging ${6*64 + 64*32 + 32*1} weights simultaneously.`, "#00ff88");
    
    model.totalSteps++;
    updateStatus(loss);
}

function renderMatrix(id, matrix, label) {
    const container = document.getElementById(id);
    if (!container) return;
    container.innerHTML = `<div class="matrix-label">${label} (${matrix.length}x${matrix[0].length})</div>`;
    
    // Scrollable wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'matrix-scroll-wrapper';
    
    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    grid.style.gridTemplateColumns = `repeat(${matrix[0].length}, 30px)`;
    
    matrix.forEach(row => {
        row.forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.width = '30px'; 
            cell.style.height = '20px';
            cell.innerText = val.toFixed(1);
            const intensity = Math.min(Math.abs(val) * 100, 100);
            if (val > 0) cell.style.background = `rgba(155, 109, 255, ${intensity/100})`;
            else cell.style.background = `rgba(255, 109, 145, ${intensity/100})`;
            grid.appendChild(cell);
        });
    });
    wrapper.appendChild(grid);
    container.appendChild(wrapper);
}

function updateVisuals() {
    renderMatrix('w1-view', model.w1, 'Weights W1 (High Res 64)');
    renderMatrix('b1-view', model.b1, 'Bias B1');
    renderMatrix('w2-view', model.w2, 'Weights W2 (High Res 32)');
    renderMatrix('b2-view', model.b2, 'Bias B2');
    renderMatrix('w3-view', model.w3, 'Weights W3');
    renderMatrix('b3-view', model.b3, 'Bias B3');
}

function updateStatus(loss) {
    document.getElementById('loss-val').innerText = loss.toFixed(6);
    document.getElementById('loss-fill').style.width = Math.max(0, 100 - (loss * 100)) + '%';
    
    const brainStatus = document.getElementById('brain-status-text');
    if (brainStatus) {
        brainStatus.innerHTML = `<strong>Status:</strong> ${model.totalSteps > 0 ? 'Trained Model' : 'Initial State'} | 
                                 <strong>Steps:</strong> ${model.totalSteps} | 
                                 <strong>Current Loss:</strong> ${loss.toFixed(6)}`;
    }

    const globalStatusText = document.getElementById('status-text');
    if (globalStatusText) {
        globalStatusText.innerText = model.totalSteps > 0 ? `Active Model (${model.totalSteps} steps)` : 'Ready / Untrained';
    }

    updateVisuals();
}

function switchMode(mode) {
    document.querySelectorAll('.mode-view').forEach(v => v.classList.add('hidden'));
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    
    if (mode === 'tutorial') {
        document.getElementById('view-tutorial').classList.remove('hidden');
        document.getElementById('nav-tut').classList.add('active');
    } else {
        document.getElementById('view-brain').classList.remove('hidden');
        document.getElementById('nav-brain').classList.add('active');
        updateVisuals(); 
    }
}

function runPrediction() {
    const n1 = parseFloat(document.getElementById('p-n1').value) || 0;
    const n2 = parseFloat(document.getElementById('p-n2').value) || 0;
    const op = document.getElementById('p-op').value;
    
    const input = [n1 / inputNorm, n2 / inputNorm, ...encodeOp(op)];
    const predRaw = model.forward(input);
    const pred = predRaw * outputNorm;
    
    document.getElementById('pred-res').innerText = pred.toFixed(2);
    
    const breakdown = document.getElementById('math-breakdown');
    breakdown.innerHTML = '';

    const steps = [
        { 
            label: 'Step 1: Input Vector',
            desc: `We convert ${n1} and ${n2} into normalized matrix form.`,
            matrix: input
        },
        { 
            label: 'Step 2: Multiply with W1',
            desc: 'Input (1x6) × W1 (6x64) → produces 64 neurons.',
            matrix: model.h1[0]
        },
        { 
            label: 'Step 3: ReLU Activation',
            desc: 'Negative values removed → neurons activated.',
            matrix: model.h1[0]
        },
        { 
            label: 'Step 4: Hidden Layer Compression',
            desc: '64 neurons → 32 features using W2.',
            matrix: model.h2[0]
        },
        { 
            label: 'Step 5: Final Output Neuron',
            desc: 'All features collapse into single value.',
            matrix: [predRaw]
        },
        { 
            label: 'Step 6: Denormalization',
            desc: `Convert back to real number scale.`,
            matrix: [pred]
        }
    ];

    steps.forEach(s => {
        const div = document.createElement('div');
        div.className = 'prediction-step';

        const title = document.createElement('strong');
        title.innerText = s.label;

        const desc = document.createElement('p');
        desc.style.fontSize = "0.75rem";
        desc.innerText = s.desc;

        div.appendChild(title);
        div.appendChild(desc);
        div.appendChild(createMatrixGrid(s.matrix, "Matrix View"));

        breakdown.appendChild(div);
    });

    // Importance Analysis
    const importanceW1 = calculateImportance(input, model.w1);
    renderMatrixWithImportance('w1-view', model.w1, 'Weights W1 (Importance Highlighted)', importanceW1);
}


function createMatrixGrid(data, label = "") {
    const container = document.createElement('div');
    container.className = 'matrix-container';

    if (label) {
        const title = document.createElement('div');
        title.className = 'matrix-label';
        title.innerText = label;
        container.appendChild(title);
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'matrix-scroll-wrapper';

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';

    // Normalize to 2D
    let matrix = Array.isArray(data[0]) ? data : [data];
    const MAX_COLS = 12;
    const displayCols = Math.min(matrix[0].length, MAX_COLS);

    grid.style.gridTemplateColumns = `repeat(${displayCols}, auto)`;

    matrix.forEach(row => {
        row.slice(0, MAX_COLS).forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.innerText = Number(val).toFixed(2);
            grid.appendChild(cell);
        });
    });

    wrapper.appendChild(grid);
    container.appendChild(wrapper);

    return container;
}

window.onload = () => { updateVisuals(); };

// ======== IMPORTANCE CALCULATION ========
function calculateImportance(input, weights) {
    let importance = [];

    for (let j = 0; j < weights[0].length; j++) {
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * weights[i][j];
        }
        importance.push(Math.abs(sum));
    }

    return importance;
}

// ======== TOP CONTRIBUTORS ========
function getTopContributors(importance, topN = 5) {
    return importance
        .map((v, i) => ({index: i, value: v}))
        .sort((a,b) => b.value - a.value)
        .slice(0, topN);
}

// ======== ENHANCED MATRIX RENDER ========
function renderMatrixWithImportance(id, matrix, label, importance = null) {
    const container = document.getElementById(id);
    if (!container) return;

    container.innerHTML = `<div class="matrix-label">${label}</div>`;

    const wrapper = document.createElement('div');
    wrapper.className = 'matrix-scroll-wrapper';

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    grid.style.gridTemplateColumns = `repeat(${matrix[0].length}, 30px)`;

    let maxImp = importance ? Math.max(...importance) : 1;

    matrix.forEach((row, r) => {
        row.forEach((val, c) => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.innerText = val.toFixed(2);

            if (importance) {
                let intensity = importance[c] / maxImp;
                cell.style.background = `rgba(255, 109, 145, ${intensity})`;
                cell.style.color = intensity > 0.6 ? 'white' : 'black';
            }

            grid.appendChild(cell);
        });
    });

    wrapper.appendChild(grid);
    container.appendChild(wrapper);
}


