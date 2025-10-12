
// pages/api/spatial-validation.js
export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        // Load spatial validation results
        const fs = require('fs');
        const path = require('path');

        const resultsPath = path.join(process.cwd(), 'outputs', 'spatial_validation_results.json');
        const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));

        res.status(200).json({
            success: true,
            data: results,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
}

// pages/api/probability-map.js
export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { lat, lon, model } = req.query;

        if (!lat || !lon) {
            return res.status(400).json({ error: 'Latitude and longitude required' });
        }

        // Load trained model and make prediction
        const prediction = await makePrediction(parseFloat(lat), parseFloat(lon), model);

        res.status(200).json({
            success: true,
            prediction: {
                latitude: parseFloat(lat),
                longitude: parseFloat(lon),
                probability: prediction.probability,
                uncertainty: prediction.uncertainty,
                model: model || 'RandomForest'
            },
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
}

// pages/api/exploration-targets.js
export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { threshold = 0.5, limit = 100 } = req.query;

        // Load exploration targets
        const targets = await getExplorationTargets(parseFloat(threshold), parseInt(limit));

        res.status(200).json({
            success: true,
            targets: targets,
            count: targets.length,
            threshold: parseFloat(threshold),
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
}

// pages/api/model-comparison.js
export default async function handler(req, res) {
    if (req.method !== 'GET') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        // Load model comparison results
        const comparison = await getModelComparison();

        res.status(200).json({
            success: true,
            comparison: comparison,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
}
