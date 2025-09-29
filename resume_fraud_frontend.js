// Grab DOM elements
const analyzeBtn = document.getElementById("analyzeBtn");
const resumeText = document.getElementById("resumeText");
const loading = document.getElementById("loading");
const resultsDiv = document.getElementById("results");
const noResultsDiv = document.getElementById("noResults");

const riskValue = document.getElementById("riskValue");
const riskLabel = document.getElementById("riskLabel");
const wordCount = document.getElementById("wordCount");
const experienceYearsEl = document.getElementById("experienceYears");
const readabilityScore = document.getElementById("readabilityScore");
const inconsistencies = document.getElementById("inconsistencies");
const flagsList = document.getElementById("flagsList");
const recommendationsList = document.getElementById("recommendationsList");

// Analyze resume function (calls backend)
async function analyzeResume() {
    const text = resumeText.value.trim();
    if (!text) {
        alert("Please paste or upload a resume first.");
        return;
    }

    loading.style.display = "block";
    resultsDiv.style.display = "none";
    noResultsDiv.style.display = "none";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ resume_text: text })
        });

        const data = await response.json();
        loading.style.display = "none";

        if (data.error) {
            alert("Error: " + data.error);
            noResultsDiv.style.display = "block";
            return;
        }

        // Display results
        resultsDiv.style.display = "block";
        riskValue.textContent = data.risk + "%";
        riskLabel.textContent = data.risk_label;
        wordCount.textContent = data.word_count;

        // Fix experience years display
        experienceYearsEl.textContent = calculateExperienceYears(text) + " years";

        readabilityScore.textContent = data.readability_score.toFixed(2);
        inconsistencies.textContent = data.inconsistencies;

        // Red flags
        flagsList.innerHTML = "";
        if (data.red_flags.length > 0) {
            data.red_flags.forEach(flag => {
                const li = document.createElement("li");
                li.textContent = flag;
                flagsList.appendChild(li);
            });
        } else {
            flagsList.innerHTML = "<li>No red flags detected</li>";
        }

        // Recommendations
        recommendationsList.innerHTML = "";
        if (data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const li = document.createElement("li");
                li.textContent = rec;
                recommendationsList.appendChild(li);
            });
        } else {
            recommendationsList.innerHTML = "<li>No recommendations</li>";
        }

    } catch (error) {
        loading.style.display = "none";
        alert("Error connecting to backend: " + error);
        noResultsDiv.style.display = "block";
    }
}

// Calculate experience years safely
function calculateExperienceYears(text) {
    const yearMatches = text.match(/\b(19|20)\d{2}\b/g) || [];
    const currentYear = new Date().getFullYear();
    const validYears = yearMatches
        .map(y => parseInt(y))
        .filter(y => y <= currentYear);

    return validYears.length > 1
        ? Math.max(...validYears) - Math.min(...validYears)
        : 0;
}

// File upload & drag-and-drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
        const reader = new FileReader();
        reader.onload = e => { resumeText.value = e.target.result; };
        reader.readAsText(file);
    } else {
        alert('Please use a .txt file or paste text directly. PDF/Word parsing is not included.');
    }
}

// Load sample resume
function loadSampleResume() {
    const sampleResume = `John Smith
Email: john.smith@email.com
Phone: 555-123-4567

Experience:
Software Engineer at Tech Corp (2020-2023)
- Developed web applications using Python and JavaScript
- Collaborated with team of 5 developers
- Built REST APIs and microservices

Junior Developer at StartUp Inc (2018-2020)
- Built web applications using React and Node.js
- Participated in agile development process
- Worked on database optimization

Education:
Bachelor of Science in Computer Science
State University (2014-2018)
- GPA: 3.7/4.0
- Relevant coursework: Data Structures, Algorithms, Database Systems

Skills: Python, JavaScript, React, Node.js, SQL, Git, AWS`;

    resumeText.value = sampleResume;
}

// Add "Load Sample Resume" button on page load
window.onload = function() {
    const inputSection = document.querySelector('.input-section');
    const sampleBtn = document.createElement('button');
    sampleBtn.textContent = '📋 Load Sample Resume';
    sampleBtn.style.cssText = `
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 20px;
        margin-top: 10px;
        cursor: pointer;
        width: 100%;
        font-size: 0.9rem;
    `;
    sampleBtn.onclick = loadSampleResume;
    inputSection.appendChild(sampleBtn);
};
