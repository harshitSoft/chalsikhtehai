import React, { useState } from 'react';
import { Box, Typography, Paper, TextField, Button, Alert } from '@mui/material';
import { QRCodeSVG } from 'qrcode.react';
import axios from 'axios';

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
            setError(err.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4, p: 2 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                    Generate User QR Code
                </Typography>
                <form onSubmit={handleSubmit}>
                    <TextField
                        fullWidth
                        label="Username"
                        name="username"
                        value={form.username}
                        onChange={handleChange}
                        margin="normal"
                        required
                    />
                    <TextField
                        fullWidth
                        label="Email"
                        name="email"
                        value={form.email}
                        onChange={handleChange}
                        margin="normal"
                        required
                        type="email"
                    />
                    <TextField
                        fullWidth
                        label="Zone"
                        name="zone"
                        value={form.zone}
                        onChange={handleChange}
                        margin="normal"
                    />
                    <TextField
                        fullWidth
                        label="Meter Number"
                        name="meter_number"
                        value={form.meter_number}
                        onChange={handleChange}
                        margin="normal"
                    />
                    <TextField
                        fullWidth
                        label="Contact Number"
                        name="contact_number"
                        value={form.contact_number}
                        onChange={handleChange}
                        margin="normal"
                    />
                    <TextField
                        fullWidth
                        label="Address"
                        name="address"
                        value={form.address}
                        onChange={handleChange}
                        margin="normal"
                    />
                    {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
                    {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
                    <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        sx={{ mt: 2 }}
                        disabled={loading}
                        fullWidth
                    >
                        {loading ? 'Generating...' : 'Generate QR'}
                    </Button>
                </form>
                {qrData && (
                    <Box sx={{ mt: 3, textAlign: 'center' }}>
                        <QRCodeSVG
                            value={JSON.stringify(qrData)}
                            size={256}
                            level="H"
                            includeMargin={true}
                        />
                        <Typography variant="body2" sx={{ mt: 2 }}>
                            Scan this QR code with the scanner to test
                        </Typography>
                    </Box>
                )}
            </Paper>
        </Box>
    );
};

export default QRGenerator; 