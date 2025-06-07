import React, { useEffect, useState, useRef } from 'react';
import { Html5QrcodeScanner } from 'html5-qrcode';
import { 
    Box, 
    Typography, 
    Paper, 
    CircularProgress, 
    Alert, 
    Card, 
    CardContent, 
    Divider,
    Button,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Grid,
    LinearProgress
} from '@mui/material';
import { PhotoCamera, Upload, Camera, QrCodeScanner } from '@mui/icons-material';
import axios from 'axios';

const QRScanner = () => {
    const [scanResult, setScanResult] = useState(null);
    const [userData, setUserData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [scannerActive, setScannerActive] = useState(true);
    const [meterReading, setMeterReading] = useState(null);
    const [captureDialogOpen, setCaptureDialogOpen] = useState(false);
    const [captureLoading, setCaptureLoading] = useState(false);
    const [captureError, setCaptureError] = useState(null);
    const [billingData, setBillingData] = useState(null);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const scannerRef = useRef(null);

    useEffect(() => {
        if (scannerActive) {
            const scanner = new Html5QrcodeScanner('reader', {
                qrbox: {
                    width: 250,
                    height: 250,
                },
                fps: 5,
                aspectRatio: 1.0,
                showTorchButtonIfSupported: true,
                showZoomSliderIfSupported: true,
            });

            scannerRef.current = scanner;

            scanner.render(onScanSuccess, onScanError);

            return () => {
                if (scannerRef.current) {
                    scannerRef.current.clear();
                }
            };
        }
    }, [scannerActive]);

    const onScanSuccess = async (decodedText) => {
        try {
            setLoading(true);
            setError(null);
            setScannerActive(false);
            
            let qrData;
            try {
                qrData = JSON.parse(decodedText);
            } catch (e) {
                // If not JSON, treat as direct username
                qrData = { username: decodedText };
            }
            
            if (!qrData.username) {
                throw new Error('Invalid QR code format: username not found');
            }

            const response = await axios.post('http://localhost:8000/scan-qr', {
                username: qrData.username
            });

            setUserData(response.data);
            setScanResult(decodedText);
        } catch (err) {
            console.error('Scan error:', err);
            if (err.response?.data?.detail) {
                setError(err.response.data.detail);
            } else if (err.message) {
                setError(err.message);
            } else {
                setError('Error scanning QR code');
            }
            setUserData(null);
            setScannerActive(true);
        } finally {
            setLoading(false);
        }
    };

    const onScanError = (error) => {
        console.warn(`QR Code scan error: ${error}`);
    };

    const handleScanAgain = () => {
        setUserData(null);
        setError(null);
        setScanResult(null);
        setMeterReading(null);
        setScannerActive(true);
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            setCaptureLoading(true);
            setCaptureError(null);
            setUploadProgress(0);

            // Validate file type
            if (!file.type.startsWith('image/')) {
                throw new Error('Please upload an image file');
            }

            // Validate file size (max 5MB)
            if (file.size > 5 * 1024 * 1024) {
                throw new Error('File size should be less than 5MB');
            }

            const formData = new FormData();
            formData.append('file', file);

            // First, get the OCR result
            const ocrResponse = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(progress);
                }
            });

            if (!ocrResponse.data.result) {
                throw new Error('No valid number detected in the image');
            }

            // Validate the reading is a valid number
            const reading = parseFloat(ocrResponse.data.result);
            if (isNaN(reading)) {
                throw new Error('Invalid reading detected in the image');
            }

            // Then, update the meter reading
            const readingResponse = await axios.post('http://localhost:8000/update-meter-reading', {
                username: userData.username,
                reading: reading
            });

            setMeterReading(reading);
            setBillingData(readingResponse.data);
            setCaptureDialogOpen(false);
            setUploadProgress(0);
        } catch (err) {
            console.error('Upload error:', err);
            setCaptureError(err.response?.data?.error || err.message || 'Error processing image');
        } finally {
            setCaptureLoading(false);
        }
    };

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
            }
        } catch (err) {
            console.error('Camera error:', err);
            setCaptureError('Error accessing camera');
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    };

    const captureImage = async () => {
        if (!videoRef.current) return;

        try {
            setCaptureLoading(true);
            setCaptureError(null);

            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('file', blob, 'capture.jpg');

                try {
                    const ocrResponse = await axios.post('http://localhost:8000/predict', formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        }
                    });

                    if (!ocrResponse.data.result) {
                        throw new Error('No valid number detected in the image');
                    }

                    const reading = parseFloat(ocrResponse.data.result);
                    if (isNaN(reading)) {
                        throw new Error('Invalid reading detected in the image');
                    }

                    const readingResponse = await axios.post('http://localhost:8000/update-meter-reading', {
                        username: userData.username,
                        reading: reading
                    });

                    setMeterReading(reading);
                    setBillingData(readingResponse.data);
                    setCaptureDialogOpen(false);
                } catch (err) {
                    console.error('Capture error:', err);
                    setCaptureError(err.response?.data?.error || err.message || 'Error processing image');
                } finally {
                    setCaptureLoading(false);
                }
            }, 'image/jpeg', 0.95);
        } catch (err) {
            console.error('Capture error:', err);
            setCaptureError('Error capturing image');
            setCaptureLoading(false);
        }
    };

    useEffect(() => {
        return () => {
            stopCamera();
            if (scannerRef.current) {
                scannerRef.current.clear();
            }
        };
    }, []);

    return (
        <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, p: 2 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                {!userData ? (
                    <Box>
                        <Typography variant="h5" gutterBottom align="center">
                            Scan QR Code
                        </Typography>
                        {error && (
                            <Alert severity="error" sx={{ mb: 2 }}>
                                {error}
                            </Alert>
                        )}
                        <Box id="reader" sx={{ width: '100%' }}></Box>
                    </Box>
                ) : (
                    <Card sx={{ mt: 3 }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                User Details
                            </Typography>
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <Typography>
                                        <strong>Username:</strong> {userData.username}
                                    </Typography>
                                </Grid>
                                <Grid item xs={12}>
                                    <Typography>
                                        <strong>Email:</strong> {userData.email}
                                    </Typography>
                                </Grid>
                            </Grid>

                            <Divider sx={{ my: 3 }} />

                            <Typography variant="h6" gutterBottom>
                                Meter Reading & Billing
                            </Typography>
                            {meterReading ? (
                                <Box sx={{ mt: 2 }}>
                                    <Typography variant="h4" color="primary" align="center" gutterBottom>
                                        Current Reading: {meterReading}
                                    </Typography>
                                    
                                    {billingData && (
                                        <Box sx={{ mt: 3 }}>
                                            <Grid container spacing={2}>
                                                <Grid item xs={6}>
                                                    <Typography variant="body1">
                                                        <strong>Last Reading:</strong> {billingData.last_unit}
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body1">
                                                        <strong>Units Consumed:</strong> {billingData.unit_consumed}
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={12}>
                                                    <Typography variant="h6" color="secondary" sx={{ mt: 2 }}>
                                                        Total Amount: ₹{billingData.total_amount}
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={12}>
                                                    <Typography variant="body2" color="text.secondary">
                                                        Last Updated: {new Date(billingData.last_reading_date).toLocaleString()}
                                                    </Typography>
                                                </Grid>
                                            </Grid>
                                        </Box>
                                    )}
                                </Box>
                            ) : (
                                <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'center' }}>
                                    <Button
                                        variant="contained"
                                        startIcon={<Camera />}
                                        onClick={() => {
                                            setCaptureDialogOpen(true);
                                            startCamera();
                                        }}
                                    >
                                        Capture Meter
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        startIcon={<Upload />}
                                        onClick={() => fileInputRef.current?.click()}
                                    >
                                        Upload Image
                                    </Button>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        hidden
                                        ref={fileInputRef}
                                        onChange={handleFileUpload}
                                    />
                                </Box>
                            )}

                            <Box sx={{ mt: 3, textAlign: 'center' }}>
                                <Typography 
                                    variant="button" 
                                    color="primary" 
                                    sx={{ cursor: 'pointer' }}
                                    onClick={handleScanAgain}
                                >
                                    Scan Another QR Code
                                </Typography>
                            </Box>
                        </CardContent>
                    </Card>
                )}
            </Paper>

            <Dialog 
                open={captureDialogOpen} 
                onClose={() => {
                    setCaptureDialogOpen(false);
                    stopCamera();
                }}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle>Capture Meter Reading</DialogTitle>
                <DialogContent>
                    <Box sx={{ mt: 2, textAlign: 'center' }}>
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            style={{ width: '100%', maxHeight: '400px' }}
                        />
                        {captureLoading && (
                            <Box sx={{ mt: 2 }}>
                                <LinearProgress variant="determinate" value={uploadProgress} />
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                    Processing... {uploadProgress}%
                                </Typography>
                            </Box>
                        )}
                        {captureError && (
                            <Typography color="error" sx={{ mt: 2 }}>
                                {captureError}
                            </Typography>
                        )}
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => {
                            setCaptureDialogOpen(false);
                            stopCamera();
                        }}
                    >
                        Cancel
                    </Button>
                    <Button 
                        onClick={captureImage}
                        variant="contained"
                        disabled={captureLoading}
                    >
                        Capture
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default QRScanner; 