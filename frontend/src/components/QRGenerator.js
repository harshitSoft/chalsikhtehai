import React, { useState, useRef } from 'react';
import { 
    Box, 
    Typography, 
    Paper, 
    TextField, 
    Button, 
    Alert,
    Grid,
    Card,
    CardContent,
    CircularProgress,
    Divider
} from '@mui/material';
import { QRCodeSVG } from 'qrcode.react';
import axios from 'axios';
import { SaveAlt as SaveAltIcon } from '@mui/icons-material';

const QRGenerator = () => {
    const [form, setForm] = useState({
        username: '',
        email: '',
        zone: '',
        meter_number: '',
        contact_number: '',
        address: ''
    });
    const [qrData, setQrData] = useState(null);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [loading, setLoading] = useState(false);
    const qrRef = useRef();

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setSuccess(null);
        setLoading(true);
        setQrData(null);
        
        try {
            // Register user
            const res = await axios.post('http://localhost:8000/users/user/register', form);
            setSuccess('User registered successfully!');
            setQrData({ ...form });
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed. Please check your inputs and try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleDownloadQR = () => {
        if (!qrRef.current) return;
        
        const svg = qrRef.current.querySelector('svg');
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svg);
        const canvas = document.createElement('canvas');
        const img = new window.Image();
        
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            const pngFile = canvas.toDataURL('image/png');
            const downloadLink = document.createElement('a');
            downloadLink.href = pngFile;
            downloadLink.download = `user_qr_${form.username || 'qr'}.png`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        };
        
        img.src = 'data:image/svg+xml;base64,' + window.btoa(unescape(encodeURIComponent(svgString)));
    };

    return (
        <Box sx={{ 
            maxWidth: 800, 
            mx: 'auto', 
            my: 4,
            px: { xs: 2, sm: 0 }
        }}>
            <Typography 
                variant="h4" 
                component="h1" 
                gutterBottom 
                sx={{ 
                    fontWeight: 600,
                    color: 'primary.main',
                    textAlign: 'center',
                    mb: 3
                }}
            >
                User Registration & QR Generator
            </Typography>
            
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                            User Information
                        </Typography>
                        <Divider sx={{ mb: 3 }} />
                        
                        <form onSubmit={handleSubmit}>
                            <Grid container spacing={2}>
                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Username"
                                        name="username"
                                        value={form.username}
                                        onChange={handleChange}
                                        margin="normal"
                                        required
                                        variant="outlined"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Email"
                                        name="email"
                                        value={form.email}
                                        onChange={handleChange}
                                        margin="normal"
                                        required
                                        type="email"
                                        variant="outlined"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Zone"
                                        name="zone"
                                        value={form.zone}
                                        onChange={handleChange}
                                        margin="normal"
                                        variant="outlined"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <TextField
                                        fullWidth
                                        label="Meter Number"
                                        name="meter_number"
                                        value={form.meter_number}
                                        onChange={handleChange}
                                        margin="normal"
                                        variant="outlined"
                                    />
                                </Grid>
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Contact Number"
                                        name="contact_number"
                                        value={form.contact_number}
                                        onChange={handleChange}
                                        margin="normal"
                                        variant="outlined"
                                    />
                                </Grid>
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Address"
                                        name="address"
                                        value={form.address}
                                        onChange={handleChange}
                                        margin="normal"
                                        multiline
                                        rows={3}
                                        variant="outlined"
                                    />
                                </Grid>
                            </Grid>
                            
                            {error && (
                                <Alert severity="error" sx={{ mt: 2 }}>
                                    {error}
                                </Alert>
                            )}
                            
                            {success && (
                                <Alert severity="success" sx={{ mt: 2 }}>
                                    {success}
                                </Alert>
                            )}
                            
                            <Button
                                type="submit"
                                variant="contained"
                                color="primary"
                                size="large"
                                sx={{ mt: 3 }}
                                disabled={loading}
                                fullWidth
                                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
                            >
                                {loading ? 'Processing...' : 'Register & Generate QR'}
                            </Button>
                        </form>
                    </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                    <Card elevation={3} sx={{ height: '100%' }}>
                        <CardContent sx={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            alignItems: 'center',
                            p: 3,
                            height: '100%'
                        }}>
                            <Typography variant="h6" gutterBottom>
                                Generated QR Code
                            </Typography>
                            <Divider sx={{ width: '100%', mb: 3 }} />
                            
                            {qrData ? (
                                <Box sx={{ 
                                    display: 'flex', 
                                    flexDirection: 'column', 
                                    alignItems: 'center',
                                    flexGrow: 1,
                                    width: '100%'
                                }} ref={qrRef}>
                                    <Box sx={{ 
                                        p: 2, 
                                        border: '1px solid #eee', 
                                        borderRadius: 1,
                                        mb: 2
                                    }}>
                                        <QRCodeSVG
                                            value={JSON.stringify(qrData)}
                                            size={256}
                                            level="H"
                                            includeMargin={true}
                                        />
                                    </Box>
                                    
                                    <Typography 
                                        variant="body2" 
                                        color="text.secondary" 
                                        sx={{ mt: 1, mb: 3 }}
                                    >
                                        Scan this QR code with the scanner application
                                    </Typography>
                                    
                                    <Button
                                        variant="contained"
                                        color="secondary"
                                        size="medium"
                                        onClick={handleDownloadQR}
                                        startIcon={<SaveAltIcon />}
                                        sx={{ mt: 'auto' }}
                                    >
                                        Download QR Code
                                    </Button>
                                </Box>
                            ) : (
                                <Box sx={{ 
                                    display: 'flex', 
                                    flexDirection: 'column', 
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexGrow: 1,
                                    width: '100%',
                                    minHeight: 300
                                }}>
                                    <Typography 
                                        variant="body1" 
                                        color="text.secondary"
                                        sx={{ textAlign: 'center' }}
                                    >
                                        {loading ? (
                                            <>
                                                <CircularProgress sx={{ mb: 2 }} />
                                                <br />
                                                Generating QR code...
                                            </>
                                        ) : (
                                            "Fill out the form and click 'Register' to generate a QR code"
                                        )}
                                    </Typography>
                                </Box>
                            )}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default QRGenerator;