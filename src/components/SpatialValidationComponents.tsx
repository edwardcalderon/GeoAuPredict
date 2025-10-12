
// components/SpatialValidationDashboard.tsx
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface SpatialValidationProps {
    model: string;
    threshold: number;
}

export const SpatialValidationDashboard: React.FC<SpatialValidationProps> = ({ 
    model, 
    threshold 
}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSpatialData();
    }, [model, threshold]);

    const fetchSpatialData = async () => {
        try {
            const response = await fetch(`/api/spatial-validation?model=${model}&threshold=${threshold}`);
            const result = await response.json();
            setData(result.data);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching spatial data:', error);
            setLoading(false);
        }
    };

    if (loading) return <div>Loading spatial validation data...</div>;

    return (
        <div className="spatial-dashboard">
            <h2>Spatial Validation Dashboard</h2>
            <div className="map-container">
                <MapContainer
                    center={[8.5, -73.0]}
                    zoom={6}
                    style={{ height: '500px', width: '100%' }}
                >
                    <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution='&copy; OpenStreetMap contributors'
                    />
                    {data?.targets?.map((target, index) => (
                        <Marker
                            key={index}
                            position={[target.lat, target.lon]}
                        >
                            <Popup>
                                <div>
                                    <h3>Exploration Target</h3>
                                    <p>Probability: {target.probability.toFixed(3)}</p>
                                    <p>Uncertainty: {target.uncertainty.toFixed(3)}</p>
                                    <p>Priority: {target.priority}</p>
                                </div>
                            </Popup>
                        </Marker>
                    ))}
                </MapContainer>
            </div>
        </div>
    );
};

// components/ModelComparison.tsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export const ModelComparison: React.FC = () => {
    const [comparisonData, setComparisonData] = useState(null);

    useEffect(() => {
        fetchModelComparison();
    }, []);

    const fetchModelComparison = async () => {
        try {
            const response = await fetch('/api/model-comparison');
            const result = await response.json();
            setComparisonData(result.comparison);
        } catch (error) {
            console.error('Error fetching model comparison:', error);
        }
    };

    return (
        <div className="model-comparison">
            <h2>Model Performance Comparison</h2>
            <ResponsiveContainer width="100%" height={400}>
                <LineChart data={comparisonData?.cv_scores}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="auc" stroke="#8884d8" name="AUC Score" />
                    <Line type="monotone" dataKey="precision_at_10" stroke="#82ca9d" name="Precision@10" />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// components/ProbabilityMap.tsx
import React, { useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

export const ProbabilityMap: React.FC = () => {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleMapClick = async (e) => {
        const { lat, lng } = e.latlng;
        setLoading(true);

        try {
            const response = await fetch(`/api/probability-map?lat=${lat}&lon=${lng}`);
            const result = await response.json();

            if (result.success) {
                setPredictions(prev => [...prev, result.prediction]);
            }
        } catch (error) {
            console.error('Error making prediction:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="probability-map">
            <h2>Interactive Probability Map</h2>
            <p>Click on the map to get gold probability predictions</p>
            <MapContainer
                center={[8.5, -73.0]}
                zoom={6}
                style={{ height: '500px', width: '100%' }}
                onClick={handleMapClick}
            >
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; OpenStreetMap contributors'
                />
                {predictions.map((pred, index) => (
                    <CircleMarker
                        key={index}
                        center={[pred.latitude, pred.longitude]}
                        radius={10}
                        color={pred.probability > 0.7 ? 'red' : pred.probability > 0.5 ? 'orange' : 'green'}
                    >
                        <Popup>
                            <div>
                                <h3>Gold Probability Prediction</h3>
                                <p>Probability: {pred.probability.toFixed(3)}</p>
                                <p>Uncertainty: {pred.uncertainty.toFixed(3)}</p>
                                <p>Model: {pred.model}</p>
                            </div>
                        </Popup>
                    </CircleMarker>
                ))}
            </MapContainer>
            {loading && <div>Making prediction...</div>}
        </div>
    );
};
