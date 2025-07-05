import React, { useEffect, useState, useRef } from 'react';
import { Html5QrcodeScanner, Html5Qrcode } from 'html5-qrcode';
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
    TextField,
    useTheme,
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
    PictureAsPdf as PdfIcon,
} from '@mui/icons-material';
import axios from 'axios';
import { useReactToPrint } from 'react-to-print';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

const QRScanner = () => {
    const theme = useTheme();
    const [scanResult, setScanResult] = useState(null);
    const [userData, setUserData] = useState(null);
    const [qrUserData, setQrUserData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [scannerActive, setScannerActive] = useState(true);
    const [meterReading, setMeterReading] = useState(null);
    const [captureDialogOpen, setCaptureDialogOpen] = useState(false);
    const [captureLoading, setCaptureLoading] = useState(false);
    const [captureError, setCaptureError] = useState(null);
    const [billingData, setBillingData] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [editableReading, setEditableReading] = useState(null);
    const [editableConsumption, setEditableConsumption] = useState(null);
    const [staffId, setStaffId] = useState('');
    const [staffName, setStaffName] = useState('');
    const [staffPromptOpen, setStaffPromptOpen] = useState(true);
    const [qrMode, setQrMode] = useState('camera');
    const qrFileInputRef = useRef(null);
    const [staffUserId, setStaffUserId] = useState(null);
    const [scannerKey, setScannerKey] = useState(0);
    const [saveLoading, setSaveLoading] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);

    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const scannerRef = useRef(null);
    const invoiceRef = useRef(null);

    const [capturedImage, setCapturedImage] = useState(null);

    // Styles
    const styles = {
        primaryCard: {
            borderRadius: theme.shape.borderRadius * 2,
            boxShadow: theme.shadows[4],
            overflow: 'hidden',
            mb: 4
        },
        sectionHeader: {
            display: 'flex',
            alignItems: 'center',
            mb: 2,
            color: theme.palette.primary.main
        },
        invoiceHeader: {
            background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
            color: theme.palette.primary.contrastText,
            p: 3,
        },
        invoiceFooter: {
            backgroundColor: theme.palette.grey[100],
            p: 2,
            textAlign: 'center'
        },
        readingCard: {
            p: 2,
            border: `1px solid ${theme.palette.grey[300]}`,
            borderRadius: theme.shape.borderRadius,
            height: '100%'
        },
        tableHeader: {
            backgroundColor: theme.palette.grey[100]
        },
        paymentInfoBox: {
            p: 2,
            backgroundColor: theme.palette.grey[50],
            borderRadius: theme.shape.borderRadius,
            border: `1px solid ${theme.palette.grey[200]}`,
            mb: 3
        },
        capturePlaceholder: {
            p: 4,
            border: `1px dashed ${theme.palette.grey[400]}`,
            borderRadius: theme.shape.borderRadius,
            textAlign: 'center'
        }
    };

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

    const handleDownloadPDF = async () => {
        if (!invoiceRef.current) return;
        
        try {
            const canvas = await html2canvas(invoiceRef.current, {
                scale: 2,
                useCORS: true,
                allowTaint: true,
                logging: false,
            });
            
            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('p', 'mm', 'a4');
            const imgWidth = 210;
            const imgHeight = (canvas.height * imgWidth) / canvas.width;
            
            pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
            pdf.save(`invoice_${generateInvoiceNumber()}.pdf`);
        } catch (error) {
            console.error('Error generating PDF:', error);
        }
    };

    useEffect(() => {
        let scanner;
        if (qrMode === 'camera' && scannerActive) {
            const readerElem = document.getElementById('reader');
            if (!readerElem) return;
            scanner = new Html5QrcodeScanner('reader', {
                qrbox: 300,
                fps: 15,
                aspectRatio: 1.333,
                showTorchButtonIfSupported: true,
                showZoomSliderIfSupported: true,
                disableFlip: true,
                experimentalFeatures: { useBarCodeDetectorIfSupported: true },
            });
            scannerRef.current = scanner;
            scanner.render(onScanSuccess, onScanError);
        }
        return () => {
            if (scannerRef.current) {
                scannerRef.current.clear().catch(() => {});
                scannerRef.current = null;
            }
        };
    // eslint-disable-next-line
    }, [qrMode, scannerActive, scannerKey]);

    useEffect(() => {
        if (meterReading && billingData) {
            setEditableReading(meterReading);
            setEditableConsumption(billingData.unit_consumed);
        }
    }, [meterReading, billingData]);

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

            setQrUserData(qrData);

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
        setQrUserData(null);
        setScannerKey(prev => prev + 1);
        setSaveSuccess(false);
    };

    const handleStaffIdSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.get(`http://localhost:8000/users/user/all`);
            const staff = res.data.find(u => String(u.created_by) === staffId && u.role === 'staff');
            if (staff) {
                setStaffName(staff.username);
                setStaffUserId(staff.id);
                setStaffPromptOpen(false);
            } else {
                setStaffName('');
                setStaffUserId(null);
                alert('Invalid Created By for Staff');
            }
        } catch {
            alert('Failed to fetch staff');
        }
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
                staff_id: staffUserId
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
                staff_id: staffUserId
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

    useEffect(() => {
        return () => stopCamera();
    }, []);

    const generateInvoiceNumber = () => {
        const date = new Date();
        return `INV-${date.getFullYear()}${(date.getMonth() + 1).toString().padStart(2, '0')}-${Math.floor(Math.random() * 9000 + 1000)}`;
    };

    const calculateDueDate = () => {
        const dueDate = new Date();
        dueDate.setDate(dueDate.getDate() + 15);
        return dueDate.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    };

    const handleReadingChange = (e) => {
        const value = parseFloat(e.target.value);
        setEditableReading(isNaN(value) ? 0 : value);
        if (billingData?.last_unit) {
            setEditableConsumption(isNaN(value) ? 0 : (value - billingData.last_unit).toFixed(2));
        }
    };

    const handleConsumptionChange = (e) => {
        const value = parseFloat(e.target.value);
        setEditableConsumption(isNaN(value) ? 0 : value);
    };

    const calculateTotalAmount = () => {
        const consumption = editableConsumption || billingData?.unit_consumed || 0;
        const baseAmount = (consumption * 12.5) + 75;
        const gst = baseAmount * 0.18;
        return (baseAmount + gst).toFixed(2);
    };

    const handleQrFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        try {
            setLoading(true);
            setError(null);
            const html5Qr = new Html5Qrcode('reader');
            const result = await html5Qr.scanFile(file, true);
            onScanSuccess(result);
        } catch (err) {
            setError('Failed to decode QR from image. Try another image.');
        } finally {
            setLoading(false);
        }
    };

    const handleSaveUnits = async () => {
        if (!userData || !editableReading || editableConsumption === null) {
            alert('Please ensure all values are filled before saving.');
            return;
        }

        try {
            setSaveLoading(true);
            setSaveSuccess(false);

            const response = await axios.post('http://localhost:8000/update-units', {
                username: userData.username,
                current_unit: editableReading,
                unit_consumed: editableConsumption,
                staff_id: staffUserId
            });

            // Update the billing data with the new values
            setBillingData(response.data);
            
            // Show success message
            setSaveSuccess(true);
            
            // Hide success message after 3 seconds
            setTimeout(() => setSaveSuccess(false), 3000);
            
        } catch (err) {
            alert(err.response?.data?.detail || 'Failed to save units. Please try again.');
        } finally {
            setSaveLoading(false);
        }
    };

    return (
        <Box sx={{ maxWidth: 'md', mx: 'auto', my: 4, p: { xs: 2, md: 3 } }}>
            {staffPromptOpen ? (
                <Paper elevation={4} sx={{ 
                    p: 3, 
                    borderRadius: theme.shape.borderRadius * 2,
                    mb: 4, 
                    textAlign: 'center',
                    maxWidth: 600,
                    mx: 'auto'
                }}>
                    <Typography variant="h5" sx={{ mb: 3 }} gutterBottom>
                        Staff Authentication
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Please enter your Staff Code to start scanning customer QR codes
                    </Typography>
                    <form onSubmit={handleStaffIdSubmit}>
                        <TextField
                            label="Staff Code"
                            value={staffId}
                            onChange={e => setStaffId(e.target.value)}
                            type="text"
                            required
                            fullWidth
                            sx={{ mb: 2 }}
                            variant="outlined"
                        />
                        <Button 
                            type="submit" 
                            variant="contained" 
                            size="large"
                            fullWidth
                        >
                            Continue
                        </Button>
                    </form>
                </Paper>
            ) : (
                <>
                    <Box sx={{ 
                        mb: 3, 
                        p: 2,
                        backgroundColor: theme.palette.primary.light,
                        color: theme.palette.primary.contrastText,
                        borderRadius: theme.shape.borderRadius,
                        textAlign: 'center'
                    }}>
                        <Typography variant="subtitle1">
                           Billed By: <strong>{staffName}</strong> 
                        </Typography>
                    </Box>

                    <Card sx={styles.primaryCard}>
                        {!userData && (
                            <Box sx={{ 
                                display: 'flex', 
                                justifyContent: 'center', 
                                gap: 2, 
                                mb: 3,
                                p: 3,
                                flexWrap: 'wrap'
                            }}>
                                <Button
                                    variant={qrMode === 'camera' ? 'contained' : 'outlined'}
                                    startIcon={<QrCodeIcon />}
                                    onClick={() => { setQrMode('camera'); setScannerActive(true); setScannerKey(prev => prev + 1); }}
                                    size="large"
                                    sx={{ minWidth: 200 }}
                                >
                                    Scan with Camera
                                </Button>
                                <Button
                                    variant={qrMode === 'upload' ? 'contained' : 'outlined'}
                                    startIcon={<UploadIcon />}
                                    onClick={() => qrFileInputRef.current?.click()}
                                    size="large"
                                    sx={{ minWidth: 200 }}
                                >
                                    Upload QR Image
                                </Button>
                                <input
                                    type="file"
                                    accept="image/*"
                                    hidden
                                    ref={qrFileInputRef}
                                    onChange={handleQrFileUpload}
                                />
                            </Box>
                        )}
                        {!userData ? (
                            <Box sx={{ p: 2 }}>
                                {qrMode === 'camera' && (
                                    <>
                                        <Box id="reader" key={scannerKey} sx={{ 
                                            width: '100%', 
                                            minHeight: 300,
                                            border: `2px dashed ${theme.palette.grey[300]}`,
                                            borderRadius: theme.shape.borderRadius
                                        }}></Box>
                                        <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
                                            Align the QR code within the frame to scan
                                        </Typography>
                                    </>
                                )}
                                {qrMode === 'upload' && (
                                    <Box sx={{ 
                                        display: 'flex', 
                                        flexDirection: 'column', 
                                        alignItems: 'center',
                                        p: 4
                                    }}>
                                        <UploadIcon sx={{ fontSize: 48, color: theme.palette.grey[400], mb: 2 }} />
                                        <Typography variant="body1" sx={{ mb: 1 }}>
                                            Upload QR Code Image
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Select an image containing the customer's QR code
                                        </Typography>
                                    </Box>
                                )}
                                {error && (
                                    <Alert severity="error" sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
                                        <WarningIcon sx={{ mr: 1 }} />
                                        {error}
                                    </Alert>
                                )}
                            </Box>
                        ) : (
                            <Box ref={invoiceRef} sx={{ backgroundColor: '#ffffff' }}>
                                {/* Invoice Header */}
                                <Box sx={styles.invoiceHeader}>
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
                                            <Typography variant="h6" sx={styles.sectionHeader}>
                                                <PersonIcon color="primary" sx={{ mr: 1 }} />
                                                Customer Details
                                            </Typography>
                                            <List dense sx={{ py: 0 }}>
                                                <ListItem sx={{ px: 0 }}>
                                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                                        <PersonIcon color="action" />
                                                    </ListItemIcon>
                                                    <ListItemText 
                                                        primary={qrUserData?.username || userData.username} 
                                                        secondary="Customer Name" 
                                                    />
                                                </ListItem>
                                                <ListItem sx={{ px: 0 }}>
                                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                                        <EmailIcon color="action" />
                                                    </ListItemIcon>
                                                    <ListItemText 
                                                        primary={qrUserData?.email || userData.email} 
                                                        secondary="Email Address" 
                                                    />
                                                </ListItem>
                                                <ListItem sx={{ px: 0 }}>
                                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                                        <HomeIcon color="action" />
                                                    </ListItemIcon>
                                                    <ListItemText 
                                                        primary={qrUserData?.address || "123, Avantika Nagar, Indore"} 
                                                        secondary="Service Address" 
                                                    />
                                                </ListItem>
                                                <ListItem sx={{ px: 0 }}>
                                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                                        <PhoneIcon color="action" />
                                                    </ListItemIcon>
                                                    <ListItemText 
                                                        primary={qrUserData?.contact_number || "+91 9876543210"} 
                                                        secondary="Contact Number" 
                                                    />
                                                </ListItem>
                                                {qrUserData?.zone && (
                                                    <ListItem sx={{ px: 0 }}>
                                                        <ListItemIcon sx={{ minWidth: 36 }}>
                                                            <HomeIcon color="action" />
                                                        </ListItemIcon>
                                                        <ListItemText 
                                                            primary={qrUserData.zone} 
                                                            secondary="Zone" 
                                                        />
                                                    </ListItem>
                                                )}
                                                {qrUserData?.meter_number && (
                                                    <ListItem sx={{ px: 0 }}>
                                                        <ListItemIcon sx={{ minWidth: 36 }}>
                                                            <DescriptionIcon color="action" />
                                                        </ListItemIcon>
                                                        <ListItemText 
                                                            primary={qrUserData.meter_number} 
                                                            secondary="Meter Number" 
                                                        />
                                                    </ListItem>
                                                )}
                                            </List>
                                        </Grid>
                                        <Grid item xs={12} md={6}>
                                            <Typography variant="h6" sx={styles.sectionHeader}>
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
                                            <Typography variant="h6" sx={styles.sectionHeader}>
                                                <GasIcon color="primary" sx={{ mr: 1 }} />
                                                Meter Reading Details
                                            </Typography>
                                            <Grid container spacing={2} sx={{ mb: 3 }}>
                                                <Grid item xs={12} md={4}>
                                                    <Box sx={styles.readingCard}>
                                                        <Typography variant="subtitle2" color="text.secondary">
                                                            Current Reading
                                                        </Typography>
                                                        <TextField
                                                            value={editableReading}
                                                            onChange={handleReadingChange}
                                                            variant="standard"
                                                            type="number"
                                                            InputProps={{
                                                                endAdornment: "units",
                                                                disableUnderline: true,
                                                                style: { fontSize: '1.5rem' }
                                                            }}
                                                            sx={{ 
                                                                '& .MuiInput-input': { 
                                                                    color: 'primary.main',
                                                                    fontWeight: 'medium'
                                                                }
                                                            }}
                                                            fullWidth
                                                            className="no-print"
                                                        />
                                                        <Typography variant="h5" color="primary" sx={{ display: ['none', 'none', 'block'] }}>
                                                            {editableReading} units
                                                        </Typography>
                                                    </Box>
                                                </Grid>
                                                <Grid item xs={12} md={4}>
                                                    <Box sx={styles.readingCard}>
                                                        <Typography variant="subtitle2" color="text.secondary">
                                                            Previous Reading
                                                        </Typography>
                                                        <Typography variant="h5">
                                                            {billingData?.last_unit || 'N/A'} units
                                                        </Typography>
                                                    </Box>
                                                </Grid>
                                                <Grid item xs={12} md={4}>
                                                    <Box sx={styles.readingCard}>
                                                        <Typography variant="subtitle2" color="text.secondary">
                                                            Consumption
                                                        </Typography>
                                                        <TextField
                                                            value={editableConsumption}
                                                            onChange={handleConsumptionChange}
                                                            variant="standard"
                                                            type="number"
                                                            InputProps={{
                                                                endAdornment: "units",
                                                                disableUnderline: true,
                                                                style: { fontSize: '1.5rem', color: 'secondary.main' }
                                                            }}
                                                            sx={{ 
                                                                '& .MuiInput-input': { 
                                                                    color: 'secondary.main',
                                                                    fontWeight: 'medium'
                                                                }
                                                            }}
                                                            fullWidth
                                                            className="no-print"
                                                        />
                                                        <Typography variant="h5" color="secondary" sx={{ display: ['none', 'none', 'block'] }}>
                                                            {editableConsumption} units
                                                        </Typography>
                                                    </Box>
                                                </Grid>
                                            </Grid>

                                            {/* Save Units Note */}
                                            {/* <Alert severity="info" sx={{ mt: 2, mb: 3 }} className="no-print">
                                                <Typography variant="body2">
                                                    <strong>Note:</strong> You can edit the Current Reading and Consumption values above. 
                                                    Use the "Save Units" button to update these values in the database for future use. 
                                                    The saved values will not affect the current invoice generation.
                                                </Typography>
                                            </Alert> */}

                                            {/* Billing Details */}
                                            <Typography variant="h6" sx={styles.sectionHeader}>
                                                <MoneyIcon color="primary" sx={{ mr: 1 }} />
                                                Billing Summary
                                            </Typography>
                                            <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 3 }}>
                                                <Table>
                                                    <TableHead sx={styles.tableHeader}>
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
                                                            <TableCell align="right">{editableConsumption || billingData?.unit_consumed || 0}</TableCell>
                                                            <TableCell align="right">12.50</TableCell>
                                                            <TableCell align="right">{(editableConsumption * 12.5 || billingData?.unit_consumed * 12.5 || 0).toFixed(2)}</TableCell>
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
                                                                {((editableConsumption * 12.5 + 75 || billingData?.unit_consumed * 12.5 + 75 || 75) * 0.18).toFixed(2)}
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
                                                                    ₹{calculateTotalAmount()}
                                                                </Typography>
                                                            </TableCell>
                                                        </TableRow>
                                                    </TableBody>
                                                </Table>
                                            </TableContainer>

                                            {/* Payment Information */}
                                            <Box sx={styles.paymentInfoBox}>
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

                                            {/* Save Units Section */}
                                            {saveSuccess && (
                                                <Alert severity="success" sx={{ mt: 2, mb: 2 }}>
                                                    Units saved successfully! The updated values are now stored in the database.
                                                </Alert>
                                            )}

                                            {/* Action Buttons */}
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                                                <Button 
                                                    variant="outlined" 
                                                    onClick={handleScanAgain} 
                                                    startIcon={<QrCodeIcon />}
                                                    className="no-print"
                                                    sx={{ mr: 2 }}
                                                >
                                                    Scan Another
                                                </Button>
                                                <Stack direction="row" spacing={2}>
                                                    <Button 
                                                        variant="contained" 
                                                        color="success"
                                                        onClick={handleSaveUnits}
                                                        disabled={saveLoading}
                                                        startIcon={<CheckCircleIcon />}
                                                        className="no-print"
                                                        sx={{ minWidth: 150 }}
                                                    >
                                                        {saveLoading ? 'Saving...' : 'Save Units'}
                                                    </Button>
                                                    <Button 
                                                        variant="contained" 
                                                        color="primary"
                                                        onClick={handlePrint} 
                                                        startIcon={<PrintIcon />}
                                                        sx={{ minWidth: 180 }}
                                                    >
                                                        Print Invoice
                                                    </Button>
                                                    <Button 
                                                        variant="contained" 
                                                        color="secondary"
                                                        onClick={handleDownloadPDF}
                                                        startIcon={<PdfIcon />}
                                                        sx={{ minWidth: 180 }}
                                                    >
                                                        Download PDF
                                                    </Button>
                                                </Stack>
                                            </Box>
                                        </Box>
                                    ) : (
                                        <Box sx={styles.capturePlaceholder}>
                                            <Avatar sx={{ 
                                                bgcolor: theme.palette.primary.main, 
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
                                <Box sx={styles.invoiceFooter}>
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
                            </Box>
                        )}
                    </Card>
                    
                    {/* Dialog for Camera Capture */}
                    <Dialog 
                        open={captureDialogOpen} 
                        onClose={() => { setCaptureDialogOpen(false); stopCamera(); }}
                        maxWidth="md"
                        fullWidth
                        PaperProps={{
                            sx: {
                                borderRadius: theme.shape.borderRadius * 2
                            }
                        }}
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
                                        borderRadius: theme.shape.borderRadius
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
                                        color: 'white',
                                        borderTopLeftRadius: theme.shape.borderRadius,
                                        borderTopRightRadius: theme.shape.borderRadius
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
                </>
            )}
        </Box>
    );
};

export default QRScanner;