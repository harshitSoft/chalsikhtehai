import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Typography,
  Box,
  Grid,
} from '@mui/material';
import Navbar from './components/Navbar';
import QRScanner from './components/QRScanner';
import QRGenerator from './components/QRGenerator';

// Theme Configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// Home Page Component
function HomePage() {
  return (
    <Box
      sx={{
        textAlign: 'center',
        mt: 4,
        py: 8,
        px: 2,
        borderRadius: '12px',
        background: 'linear-gradient(-45deg, #1d3557, #457b9d, #a8dadc, #f1faee)',
        backgroundSize: '400% 400%',
        animation: 'gradientBG 15s ease infinite',
        color: 'white',
      }}
    >
      <Typography variant="h3" gutterBottom sx={{ fontWeight: 700 }}>
        Welcome to Meter Reader
      </Typography>
      <Typography variant="h6" sx={{ opacity: 0.9 }}>
        Scan and Generate QR Codes for Quick User Access
      </Typography>

      <Grid container spacing={4} justifyContent="center" mt={4}>
        <Grid item>
          <Link to="/scan">
            <img
              src="https://wpblogassets.paytm.com/paytmblog/uploads/2022/01/3_What-is-a-QR-code-Know-everything-about-it-800x500.jpg"
              alt="Scan QR"
              style={{
                width: '250px',
                height: '150px',
                objectFit: 'cover',
                borderRadius: '12px',
                boxShadow: '0 4px 10px rgba(0,0,0,0.2)',
                transition: 'transform 0.3s ease',
              }}
              onMouseOver={e => (e.currentTarget.style.transform = 'scale(1.05)')}
              onMouseOut={e => (e.currentTarget.style.transform = 'scale(1)')}
            />
          </Link>
          <Typography variant="subtitle1" mt={1} color="white">
            Scan QR
          </Typography>
        </Grid>
        <Grid item>
          <Link to="/generate">
            <img
              src="https://www.joydeepdeb.com/images/qr-code.jpg"
              alt="Generate QR"
              style={{
                width: '250px',
                height: '150px',
                objectFit: 'cover',
                borderRadius: '12px',
                boxShadow: '0 4px 10px rgba(0,0,0,0.2)',
                transition: 'transform 0.3s ease',
              }}
              onMouseOver={e => (e.currentTarget.style.transform = 'scale(1.05)')}
              onMouseOut={e => (e.currentTarget.style.transform = 'scale(1)')}
            />
          </Link>
          <Typography variant="subtitle1" mt={1} color="white">
            Generate QR
          </Typography>
        </Grid>
      </Grid>
    </Box>
  );
}

// App Component
function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Container maxWidth="lg">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/scan" element={<QRScanner />} />
            <Route path="/generate" element={<QRGenerator />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;
