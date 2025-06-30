import React, { useEffect, useState, useRef } from 'react';
import { Html5QrcodeScanner } from 'html5-qrcode';
import {
    Box,
    Typography,
    Paper,
    Alert,
    Card,
    CardContent,
    Divider,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Grid,
    LinearProgress,
    Chip,
    Avatar,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Stack,
} from '@mui/material';
import {
    Upload as UploadIcon,
    CameraAlt as CameraIcon,
    Print as PrintIcon,
    CheckCircle as CheckCircleIcon,
    Receipt as ReceiptIcon,
    Person as PersonIcon,
    Email as EmailIcon,
    Home as HomeIcon,
    Phone as PhoneIcon,
    Event as EventIcon,
    LocalGasStation as GasIcon,
    AttachMoney as MoneyIcon,
    Description as DescriptionIcon,
    AccountBalance as BankIcon,
    Warning as WarningIcon,
    QrCodeScanner as QrCodeIcon,
} from '@mui/icons-material';
import axios from 'axios';
import { useReactToPrint } from 'react-to-print';

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
    const [uploadProgress, setUploadProgress] = useState(0);

    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const scannerRef = useRef(null);
    const invoiceRef = useRef(null);

    const handlePrint = useReactToPrint({
        content: () => invoiceRef.current,
        pageStyle: `
            @page { size: A4; margin: 10mm; }
            @media print {
                body { -webkit-print-color-adjust: exact; }
                .no-print { display: none !important; }
            }
        `,
    });

    useEffect(() => {
        if (scannerActive) {
            const scanner = new Html5QrcodeScanner('reader', {
                qrbox: { width: 250, height: 250 },
                fps: 5,
                aspectRatio: 1.0,
                showTorchButtonIfSupported: true,
                showZoomSliderIfSupported: true,
            });
            scannerRef.current = scanner;
            scanner.render(onScanSuccess, onScanError);
            return () => scanner.clear();
        }
    }, [scannerActive]);

    const onScanSuccess = async (decodedText) => {
        try {
            setLoading(true);
            setError(null);
            setScannerActive(false);

            let qrData = { username: decodedText };
            try {
                qrData = JSON.parse(decodedText);
            } catch {}

            if (!qrData.username) throw new Error('Invalid QR code format');

            const response = await axios.post('http://localhost:8000/scan-qr', { username: qrData.username });
            setUserData(response.data);
            setScanResult(decodedText);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'QR scan error');
            setUserData(null);
            setScannerActive(true);
        } finally {
            setLoading(false);
        }
    };

    const onScanError = (err) => console.warn('Scan error', err);

    const handleScanAgain = () => {
        setUserData(null);
        setScanResult(null);
        setMeterReading(null);
        setBillingData(null);
        setError(null);
        setScannerActive(true);
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            setCaptureLoading(true);
            setUploadProgress(0);

            if (!file.type.startsWith('image/') || file.size > 5 * 1024 * 1024) {
                throw new Error('Please upload an image file less than 5MB');
            }

            const formData = new FormData();
            formData.append('file', file);

            const ocrRes = await axios.post('http://localhost:8000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (e) => setUploadProgress(Math.round((e.loaded * 100) / e.total)),
            });

            const reading = parseFloat(ocrRes.data.result);
            if (isNaN(reading)) throw new Error('Invalid meter reading detected');

            const billingRes = await axios.post('http://localhost:8000/update-meter-reading', {
                username: userData.username,
                reading,
            });

            setMeterReading(reading);
            setBillingData(billingRes.data);
            setCaptureDialogOpen(false);
        } catch (err) {
            setCaptureError(err.message || 'Image processing failed');
        } finally {
            setCaptureLoading(false);
        }
    };

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            if (videoRef.current) videoRef.current.srcObject = stream;
            streamRef.current = stream;
        } catch (err) {
            setCaptureError('Camera access denied. Please allow camera permissions.');
        }
    };

    const stopCamera = () => {
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
    };

    const captureImage = async () => {
        if (!videoRef.current) return;
        setCaptureLoading(true);
        setCaptureError(null);

        try {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
            const formData = new FormData();
            formData.append('file', blob, 'meter_capture.jpg');

            const ocrRes = await axios.post('http://localhost:8000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const reading = parseFloat(ocrRes.data.result);
            if (isNaN(reading)) throw new Error('Invalid meter reading detected');

            const billingRes = await axios.post('http://localhost:8000/update-meter-reading', {
                username: userData.username,
                reading,
            });

            setMeterReading(reading);
            setBillingData(billingRes.data);
            setCaptureDialogOpen(false);
        } catch (err) {
            setCaptureError(err.message || 'Failed to process meter reading');
        } finally {
            setCaptureLoading(false);
        }
    };

    useEffect(() => () => stopCamera(), []);

    const generateInvoiceNumber = () => {
        const date = new Date();
        return `INV-${date.getFullYear()}${(date.getMonth() + 1).toString().padStart(2, '0')}-${Math.floor(Math.random() * 9000 + 1000)}`;
    };

    const calculateDueDate = () => {
        const dueDate = new Date();
        dueDate.setDate(dueDate.getDate() + 15);
        return dueDate.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    };

    return (
        <Box sx={{ maxWidth: 'md', mx: 'auto', my: 4, p: { xs: 1, md: 3 } }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
                <Typography variant="h3" sx={{ 
                    fontWeight: 'bold', 
                    background: 'linear-gradient(to right, #1976d2, #4caf50)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    display: 'inline-block',
                    mb: 1
                }}>
                    Avantika Gas Services
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                    Digital Meter Reading & Billing Portal
                </Typography>
            </Box>

            <Paper elevation={4} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
                {!userData ? (
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
                            <QrCodeIcon color="primary" sx={{ fontSize: 32, mr: 1 }} />
                            <Typography variant="h5" align="center">
                                Scan Customer QR Code
                            </Typography>
                        </Box>
                        {error && (
                            <Alert severity="error" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                                <WarningIcon sx={{ mr: 1 }} />
                                {error}
                            </Alert>
                        )}
                        <Box id="reader" sx={{ width: '100%', minHeight: 300 }}></Box>
                        <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
                            Point your camera at the customer's QR code to begin
                        </Typography>
                    </Box>
                ) : (
                    <Card ref={invoiceRef} sx={{ backgroundColor: '#ffffff', p: 0, border: '1px solid #e0e0e0' }}>
                        {/* Invoice Header */}
                        <Box sx={{ 
                            backgroundColor: '#1976d2', 
                            color: 'white', 
                            p: 3, 
                            borderTopLeftRadius: 4,
                            borderTopRightRadius: 4,
                        }}>
                            <Grid container spacing={2} alignItems="center">
                                <Grid item xs={12} md={6}>
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                        <GasIcon sx={{ fontSize: 40, mr: 2 }} />
                                        <Box>
                                            <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                                                AVANTIKA GAS
                                            </Typography>
                                            <Typography variant="body2">
                                                Energy for Sustainable Living
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Grid>
                                <Grid item xs={12} md={6} sx={{ textAlign: { xs: 'left', md: 'right' } }}>
                                    <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                                        INVOICE
                                    </Typography>
                                    <Typography variant="body2">
                                        {new Date().toLocaleDateString('en-IN', { 
                                            day: 'numeric', 
                                            month: 'long', 
                                            year: 'numeric' 
                                        })}
                                    </Typography>
                                </Grid>
                            </Grid>
                        </Box>

                        <CardContent sx={{ p: 3 }}>
                            {/* Customer Information */}
                            <Grid container spacing={3} sx={{ mb: 3 }}>
                                <Grid item xs={12} md={6}>
                                    <Typography variant="h6" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                                        <PersonIcon color="primary" sx={{ mr: 1 }} />
                                        Customer Details
                                    </Typography>
                                    <List dense sx={{ py: 0 }}>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <PersonIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary={userData.username} 
                                                secondary="Customer Name" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <EmailIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary={userData.email} 
                                                secondary="Email Address" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <HomeIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary="123, Avantika Nagar, Indore" 
                                                secondary="Service Address" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <PhoneIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary="+91 9876543210" 
                                                secondary="Contact Number" 
                                            />
                                        </ListItem>
                                    </List>
                                </Grid>
                                <Grid item xs={12} md={6}>
                                    <Typography variant="h6" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                                        <ReceiptIcon color="primary" sx={{ mr: 1 }} />
                                        Invoice Information
                                    </Typography>
                                    <List dense sx={{ py: 0 }}>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <DescriptionIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary={generateInvoiceNumber()} 
                                                secondary="Invoice Number" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <EventIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary={new Date().toLocaleDateString('en-IN', { 
                                                    day: 'numeric', 
                                                    month: 'short', 
                                                    year: 'numeric' 
                                                })} 
                                                secondary="Invoice Date" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <EventIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary={calculateDueDate()} 
                                                secondary="Due Date" 
                                            />
                                        </ListItem>
                                        <ListItem sx={{ px: 0 }}>
                                            <ListItemIcon sx={{ minWidth: 36 }}>
                                                <CheckCircleIcon color="action" />
                                            </ListItemIcon>
                                            <ListItemText 
                                                primary="Online Payment" 
                                                secondary="Payment Method" 
                                            />
                                        </ListItem>
                                    </List>
                                </Grid>
                            </Grid>

                            {/* Meter Reading Section */}
                            {meterReading ? (
                                <Box sx={{ mb: 4 }}>
                                    <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                                        <GasIcon color="primary" sx={{ mr: 1 }} />
                                        Meter Reading Details
                                    </Typography>
                                    <Grid container spacing={2} sx={{ mb: 3 }}>
                                        <Grid item xs={12} md={4}>
                                            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Current Reading
                                                </Typography>
                                                <Typography variant="h5" color="primary">
                                                    {meterReading} units
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                        <Grid item xs={12} md={4}>
                                            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Previous Reading
                                                </Typography>
                                                <Typography variant="h5">
                                                    {billingData?.last_unit || 'N/A'} units
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                        <Grid item xs={12} md={4}>
                                            <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Consumption
                                                </Typography>
                                                <Typography variant="h5" color="secondary">
                                                    {billingData?.unit_consumed || 'N/A'} units
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                    </Grid>

                                    {/* Billing Details */}
                                    <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                                        <MoneyIcon color="primary" sx={{ mr: 1 }} />
                                        Billing Summary
                                    </Typography>
                                    <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 3 }}>
                                        <Table>
                                            <TableHead sx={{ backgroundColor: '#f5f5f5' }}>
                                                <TableRow>
                                                    <TableCell>Description</TableCell>
                                                    <TableCell align="right">Units</TableCell>
                                                    <TableCell align="right">Rate (₹)</TableCell>
                                                    <TableCell align="right">Amount (₹)</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>Gas Consumption Charges</TableCell>
                                                    <TableCell align="right">{billingData?.unit_consumed || 0}</TableCell>
                                                    <TableCell align="right">12.50</TableCell>
                                                    <TableCell align="right">{(billingData?.unit_consumed * 12.5).toFixed(2)}</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>Fixed Monthly Charges</TableCell>
                                                    <TableCell align="right">-</TableCell>
                                                    <TableCell align="right">-</TableCell>
                                                    <TableCell align="right">75.00</TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>GST (18%)</TableCell>
                                                    <TableCell align="right">-</TableCell>
                                                    <TableCell align="right">-</TableCell>
                                                    <TableCell align="right">
                                                        {((billingData?.unit_consumed * 12.5 + 75) * 0.18).toFixed(2)}
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow sx={{ '&:last-child td': { borderBottom: 0 } }}>
                                                    <TableCell colSpan={3} align="right">
                                                        <Typography variant="subtitle1">
                                                            Total Payable
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        <Typography variant="h6" color="primary">
                                                            ₹{billingData?.total_amount || '0.00'}
                                                        </Typography>
                                                    </TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>

                                    {/* Payment Information */}
                                    <Box sx={{ p: 2, backgroundColor: '#f9f9f9', borderRadius: 1, mb: 3 }}>
                                        <Typography variant="subtitle1" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                                            <BankIcon color="primary" sx={{ mr: 1 }} />
                                            Payment Information
                                        </Typography>
                                        <Typography variant="body2" sx={{ mb: 1 }}>
                                            Please make payment to the following account:
                                        </Typography>
                                        <List dense>
                                            <ListItem sx={{ px: 0 }}>
                                                <ListItemText 
                                                    primary="Avantika Gas Services Ltd." 
                                                    secondary="Account Name" 
                                                />
                                            </ListItem>
                                            <ListItem sx={{ px: 0 }}>
                                                <ListItemText 
                                                    primary="AXIS0001234" 
                                                    secondary="Account Number" 
                                                />
                                            </ListItem>
                                            <ListItem sx={{ px: 0 }}>
                                                <ListItemText 
                                                    primary="UTIB0000123" 
                                                    secondary="IFSC Code" 
                                                />
                                            </ListItem>
                                        </List>
                                        <Typography variant="body2" color="error.main" sx={{ mt: 1 }}>
                                            Note: Please include your invoice number as payment reference.
                                        </Typography>
                                    </Box>

                                    {/* Action Buttons */}
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                                        <Button 
                                            variant="outlined" 
                                            onClick={handleScanAgain} 
                                            startIcon={<QrCodeIcon />}
                                            className="no-print"
                                        >
                                            Scan Another
                                        </Button>
                                        <Stack direction="row" spacing={2}>
                                            <Button 
                                                variant="contained" 
                                                color="primary"
                                                onClick={handlePrint} 
                                                startIcon={<PrintIcon />}
                                            >
                                                Print Invoice
                                            </Button>
                                            <Button 
                                                variant="contained" 
                                                color="success"
                                                className="no-print"
                                            >
                                                Make Payment
                                            </Button>
                                        </Stack>
                                    </Box>
                                </Box>
                            ) : (
                                <Box sx={{ textAlign: 'center', p: 4, border: '1px dashed #e0e0e0', borderRadius: 1 }}>
                                    <Avatar sx={{ 
                                        bgcolor: '#1976d2', 
                                        width: 56, 
                                        height: 56, 
                                        mb: 2,
                                        mx: 'auto'
                                    }}>
                                        <GasIcon fontSize="large" />
                                    </Avatar>
                                    <Typography variant="h6" gutterBottom>
                                        Capture Meter Reading
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                                        Please capture or upload an image of your gas meter to generate the invoice
                                    </Typography>
                                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
                                        <Button 
                                            variant="contained" 
                                            startIcon={<CameraIcon />} 
                                            onClick={() => { setCaptureDialogOpen(true); startCamera(); }}
                                            size="large"
                                        >
                                            Use Camera
                                        </Button>
                                        <Button 
                                            variant="outlined" 
                                            startIcon={<UploadIcon />} 
                                            onClick={() => fileInputRef.current?.click()}
                                            size="large"
                                        >
                                            Upload Image
                                        </Button>
                                        <input 
                                            type="file" 
                                            hidden 
                                            accept="image/*" 
                                            ref={fileInputRef} 
                                            onChange={handleFileUpload} 
                                        />
                                    </Stack>
                                </Box>
                            )}
                        </CardContent>

                        {/* Invoice Footer */}
                        <Box sx={{ 
                            backgroundColor: '#f5f5f5', 
                            p: 2, 
                            textAlign: 'center',
                            borderBottomLeftRadius: 4,
                            borderBottomRightRadius: 4,
                        }}>
                            <Typography variant="body2" color="text.secondary">
                                Thank you for choosing Avantika Gas Services
                            </Typography>
                            <Typography variant="caption" display="block" color="text.secondary">
                                For any queries, please contact customer support at support@avantikagas.com or call 1800-123-4567
                            </Typography>
                            <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
                                This is a computer generated invoice and does not require signature
                            </Typography>
                        </Box>
                    </Card>
                )}
            </Paper>

            {/* Camera Capture Dialog */}
            <Dialog 
                open={captureDialogOpen} 
                onClose={() => { setCaptureDialogOpen(false); stopCamera(); }}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
                    <CameraIcon color="primary" sx={{ mr: 1 }} />
                    Live Meter Capture
                </DialogTitle>
                <DialogContent>
                    <Box sx={{ position: 'relative', height: 400 }}>
                        <video 
                            ref={videoRef} 
                            autoPlay 
                            playsInline 
                            style={{ 
                                width: '100%', 
                                height: '100%', 
                                objectFit: 'contain',
                                backgroundColor: '#000',
                                borderRadius: 4
                            }} 
                        />
                        {captureLoading && (
                            <Box sx={{ 
                                position: 'absolute', 
                                top: 0, 
                                left: 0, 
                                right: 0, 
                                p: 2,
                                backgroundColor: 'rgba(0,0,0,0.5)',
                                color: 'white'
                            }}>
                                <LinearProgress variant="determinate" value={uploadProgress} color="primary" />
                                <Typography variant="body2" align="center" sx={{ mt: 1 }}>
                                    Processing meter reading... {uploadProgress}%
                                </Typography>
                            </Box>
                        )}
                    </Box>
                    {captureError && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                            {captureError}
                        </Alert>
                    )}
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                        Position the meter within the frame and ensure the numbers are clearly visible
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => { setCaptureDialogOpen(false); stopCamera(); }}
                        color="secondary"
                    >
                        Cancel
                    </Button>
                    <Button 
                        variant="contained" 
                        onClick={captureImage} 
                        disabled={captureLoading}
                        startIcon={<CameraIcon />}
                    >
                        Capture Reading
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default QRScanner;