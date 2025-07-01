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
  Button,
  Card,
  CardContent,
  CardActionArea,
  CardMedia,
  Divider,
  Stack,
  Paper,
  useMediaQuery,
} from '@mui/material';
import {
  QrCodeScanner as ScanIcon,
  QrCode as GenerateIcon,
  LocalGasStation as GasIcon,
  Speed as SpeedIcon,
  Receipt as BillIcon,
  Security as SecurityIcon,
  SupportAgent as SupportIcon,
} from '@mui/icons-material';
import Navbar from './components/Navbar';
import QRScanner from './components/QRScanner';
import QRGenerator from './components/QRGenerator';
import AdminPanel from './components/AdminPanel';
import UserSectionTable from './components/UserList';

// Theme Configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1a5276',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#e67e22',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f9f9f9',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 500,
      fontSize: '1.75rem',
    },
  },
});

// Feature Card Component
const FeatureCard = ({ icon, title, description }) => (
  <Card sx={{ height: '100%', borderRadius: 2, boxShadow: 3 }}>
    <CardContent sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {React.cloneElement(icon, { color: 'primary', sx: { fontSize: 40, mr: 2 } })}
        <Typography variant="h5" component="h3">
          {title}
        </Typography>
      </Box>
      <Typography variant="body1" color="text.secondary">
        {description}
      </Typography>
    </CardContent>
  </Card>
);

// Action Card Component
const ActionCard = ({ to, icon, title, description, image }) => (
  <Card sx={{ height: '100%', borderRadius: 2, boxShadow: 3 }}>
    <CardActionArea component={Link} to={to} sx={{ height: '100%' }}>
      <CardMedia
        component="img"
        height="160"
        image={image}
        alt={title}
        sx={{ objectFit: 'cover' }}
      />
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          {React.cloneElement(icon, { color: 'primary', sx: { fontSize: 30, mr: 2 } })}
          <Typography variant="h5" component="h3">
            {title}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </CardContent>
    </CardActionArea>
  </Card>
);

// Hero Section Component
const HeroSection = () => {
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box
      sx={{
        position: 'relative',
        height: isSmallScreen ? '60vh' : '70vh',
        backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url(https://images.unsplash.com/photo-1606229365485-93a3b8ee0385?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80)',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        px: 2,
        mb: 6,
      }}
    >
      <Container maxWidth="md">
        <Typography variant="h1" component="h1" gutterBottom>
          Avantika Gas Services
        </Typography>
        <Typography variant="h4" component="h2" gutterBottom sx={{ mb: 4 }}>
          Digital Meter Reading & Billing Solution
        </Typography>
        <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
          Fast, secure, and contactless gas billing through QR technology
        </Typography>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
          <Button
            variant="contained"
            color="secondary"
            size="large"
            component={Link}
            to="/scan"
            startIcon={<ScanIcon />}
            sx={{ px: 4, py: 1.5 }}
          >
            Scan QR Code
          </Button>
          <Button
            variant="outlined"
            color="inherit"
            size="large"
            component={Link}
            to="/generate"
            startIcon={<GenerateIcon />}
            sx={{ px: 4, py: 1.5 }}
          >
            Generate QR
          </Button>
        </Stack>
      </Container>
    </Box>
  );
};

// Home Page Component
function HomePage() {
  return (
    <Box>
      <HeroSection />

      <Container maxWidth="lg" sx={{ my: 8 }}>
        {/* Features Section */}
        <Typography variant="h2" component="h2" align="center" gutterBottom sx={{ mb: 6 }}>
          Why Choose Our Digital Solution
        </Typography>
        <Grid container spacing={4} sx={{ mb: 10 }}>
          <Grid item xs={12} md={4}>
            <FeatureCard
              icon={<SpeedIcon />}
              title="Instant Processing"
              description="Generate bills in seconds by simply scanning your meter QR code, eliminating long wait times."
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard
              icon={<BillIcon />}
              title="Paperless Billing"
              description="Go green with our digital invoicing system that reduces paper waste and keeps records organized."
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard
              icon={<SecurityIcon />}
              title="Secure Transactions"
              description="Bank-grade encryption ensures all your billing data and payments remain completely secure."
            />
          </Grid>
        </Grid>

        {/* How It Works Section */}
        <Paper elevation={0} sx={{ p: 4, mb: 10, borderRadius: 3, backgroundColor: '#f5f5f5' }}>
          <Typography variant="h2" component="h2" align="center" gutterBottom sx={{ mb: 6 }}>
            How It Works
          </Typography>
          <Grid container spacing={4}>
            {[
              {
                step: '1',
                title: 'Generate Your QR Code',
                description: 'Each customer receives a unique QR code linked to their gas account.',
              },
              {
                step: '2',
                title: 'Scan with Mobile Device',
                description: 'Field agents scan the QR code using our secure mobile application.',
              },
              {
                step: '3',
                title: 'Capture Meter Reading',
                description: 'Take a photo of your gas meter or enter the reading manually.',
              },
              {
                step: '4',
                title: 'Instant Bill Generation',
                description: 'The system automatically calculates your bill with current rates.',
              },
            ].map((item, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Box sx={{ textAlign: 'center', height: '100%' }}>
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      backgroundColor: 'primary.main',
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '1.5rem',
                      fontWeight: 'bold',
                      mx: 'auto',
                      mb: 2,
                    }}
                  >
                    {item.step}
                  </Box>
                  <Typography variant="h5" gutterBottom>
                    {item.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {item.description}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Action Cards Section */}
        <Typography variant="h2" component="h2" align="center" gutterBottom sx={{ mb: 6 }}>
          Get Started
        </Typography>
        <Grid container spacing={4} sx={{ mb: 8 }}>
          <Grid item xs={12} md={6}>
            <ActionCard
              to="/scan"
              icon={<ScanIcon />}
              title="Scan QR Code"
              description="Scan customer QR codes to quickly access accounts and record meter readings."
              image="https://t3.ftcdn.net/jpg/04/73/03/74/360_F_473037464_kPgGJRfT3GxL45eI9NQDNpR3xovJjYLc.jpg"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <ActionCard
              to="/generate"
              icon={<GenerateIcon />}
              title="Generate QR"
              description="Create new QR codes for customer accounts and meter installations."
              image="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStEW4MISAdx0ks98OJieIbhspqzlPqfy87IA&s"
            />
          </Grid>
        </Grid>

        {/* Support Section */}
        <Paper elevation={0} sx={{ p: 4, borderRadius: 3, backgroundColor: 'primary.main', color: 'white' }}>
          <Grid container alignItems="center" spacing={4}>
            <Grid item xs={12} md={8}>
              <Typography variant="h3" component="h3" gutterBottom>
                Need Help With Our System?
              </Typography>
              <Typography variant="body1" sx={{ mb: 3, opacity: 0.9 }}>
                Our customer support team is available 24/7 to assist you with any questions about QR code scanning, bill generation, or account management.
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                size="large"
                startIcon={<SupportIcon />}
                sx={{ px: 4 }}
              >
                Contact Support
              </Button>
            </Grid>
            <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'center' }}>
              <SupportIcon sx={{ fontSize: 120, opacity: 0.2 }} />
            </Grid>
          </Grid>
        </Paper>
      </Container>
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
            <Route path="/users" element={<UserSectionTable />} />
            <Route path="/admin-panel" element={<AdminPanel />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;