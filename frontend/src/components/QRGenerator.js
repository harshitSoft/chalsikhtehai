import React, { useState } from 'react';
import { Box, Typography, Paper, TextField, Button } from '@mui/material';
import { QRCodeSVG } from 'qrcode.react';

const QRGenerator = () => {
    const [username, setUsername] = useState('');

    const qrValue = username ? JSON.stringify({ username }) : '';

    return (
        <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4, p: 2 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                    Generate Test QR Code
                </Typography>

                <TextField
                    fullWidth
                    label="Enter Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    margin="normal"
                />

                {qrValue && (
                    <Box sx={{ mt: 3, textAlign: 'center' }}>
                        <QRCodeSVG
                            value={qrValue}
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