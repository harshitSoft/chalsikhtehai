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
  useTheme,
  alpha,
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

// Enhanced Theme Configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1a5276',
      light: '#4a7fa8',
      dark: '#002946',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#e67e22',
      light: '#ffae4a',
      dark: '#ad5100',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
    },
  },
  typography: {
    fontFamily: '"Inter", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      fontSize: '3.5rem',
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.3,
      letterSpacing: '-0.015em',
    },
    h3: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.5,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.6,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.7,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.7,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '10px 24px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          transition: 'transform 0.2s, box-shadow 0.2s',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 10px 20px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
        },
      },
    },
  },
});

// Feature Card Component
const FeatureCard = ({ icon, title, description }) => {
  const theme = useTheme();
  
  return (
    <Card sx={{ 
      height: '100%', 
      border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
      background: theme.palette.background.paper,
    }}>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          mb: 2,
          color: theme.palette.primary.main,
        }}>
          {React.cloneElement(icon, { sx: { fontSize: 40, mr: 2 } })}
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
};

// Action Card Component
const ActionCard = ({ to, icon, title, description, image }) => {
  const theme = useTheme();
  
  return (
    <Card sx={{ 
      height: '100%',
      background: theme.palette.background.paper,
    }}>
      <CardActionArea 
        component={Link} 
        to={to} 
        sx={{ 
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-start',
        }}
      >
        <CardMedia
          component="img"
          height="180"
          image={image}
          alt={title}
          sx={{ 
            objectFit: 'cover',
            borderTopLeftRadius: '12px',
            borderTopRightRadius: '12px',
          }}
        />
        <CardContent sx={{ p: 3, flexGrow: 1 }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 2,
            color: theme.palette.primary.main,
          }}>
            {React.cloneElement(icon, { sx: { fontSize: 30, mr: 2 } })}
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
};

// Hero Section Component
const HeroSection = () => {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box
      sx={{
        position: 'relative',
        height: isSmallScreen ? '60vh' : '80vh',
        minHeight: '500px',
        backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url(https://images.unsplash.com/photo-1606229365485-93a3b8ee0385?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80)',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        px: 2,
        mb: 10,
        '&::before': {
          content: '""',
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '100px',
          background: `linear-gradient(to bottom, transparent, ${theme.palette.background.default})`,
          zIndex: 1,
        },
      }}
    >
      <Container maxWidth="md" sx={{ position: 'relative', zIndex: 2 }}>
        <Typography 
          variant="h1" 
          component="h1" 
          gutterBottom
          sx={{
            textShadow: '0 2px 4px rgba(0,0,0,0.3)',
            mb: 3,
          }}
        >
          Avantika Gas Services
        </Typography>
        <Typography 
          variant="h3" 
          component="h2" 
          gutterBottom 
          sx={{ 
            mb: 4,
            fontWeight: 500,
            textShadow: '0 2px 4px rgba(0,0,0,0.2)',
          }}
        >
          Digital Meter Reading & Billing Solution
        </Typography>
        <Typography 
          variant="h6" 
          sx={{ 
            mb: 5, 
            opacity: 0.9,
            maxWidth: '700px',
            mx: 'auto',
            textShadow: '0 1px 2px rgba(0,0,0,0.2)',
          }}
        >
          Revolutionizing gas billing with secure, fast, and contactless QR technology solutions
        </Typography>
        <Stack 
          direction={{ xs: 'column', sm: 'row' }} 
          spacing={3} 
          justifyContent="center"
          sx={{ maxWidth: '600px', mx: 'auto' }}
        >
          <Button
            variant="contained"
            color="secondary"
            size="large"
            component={Link}
            to="/scan"
            startIcon={<ScanIcon />}
            sx={{ 
              px: 5,
              py: 1.5,
              fontSize: '1.1rem',
            }}
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
            sx={{ 
              px: 5,
              py: 1.5,
              fontSize: '1.1rem',
              borderWidth: '2px',
              '&:hover': {
                borderWidth: '2px',
              },
            }}
          >
            Generate QR
          </Button>
        </Stack>
      </Container>
    </Box>
  );
};

// Step Component for How It Works section
const Step = ({ number, title, description }) => {
  const theme = useTheme();
  
  return (
    <Box sx={{ textAlign: 'center', height: '100%', px: 2 }}>
      <Box
        sx={{
          width: 60,
          height: 60,
          borderRadius: '50%',
          backgroundColor: alpha(theme.palette.primary.main, 0.1),
          color: theme.palette.primary.main,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1.5rem',
          fontWeight: 'bold',
          mx: 'auto',
          mb: 3,
          border: `2px solid ${alpha(theme.palette.primary.main, 0.2)}`,
        }}
      >
        {number}
      </Box>
      <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
        {title}
      </Typography>
      <Typography variant="body1" color="text.secondary">
        {description}
      </Typography>
    </Box>
  );
};

// Home Page Component
function HomePage() {
  const theme = useTheme();
  
  return (
    <Box sx={{ overflowX: 'hidden' }}>
      <HeroSection />

      <Container maxWidth="xl" sx={{ my: 10 }}>
        {/* Features Section */}
        <Box sx={{ mb: 15 }}>
          <Typography 
            variant="h2" 
            component="h2" 
            align="center" 
            gutterBottom 
            sx={{ 
              mb: 8,
              position: 'relative',
              '&::after': {
                content: '""',
                display: 'block',
                width: '80px',
                height: '4px',
                backgroundColor: theme.palette.secondary.main,
                margin: '20px auto 0',
                borderRadius: '2px',
              },
            }}
          >
            Why Choose Our Digital Solution
          </Typography>
          <Grid container spacing={5} sx={{ px: { xs: 0, md: 5 } }}>
            <Grid item xs={12} md={4}>
              <FeatureCard
                icon={<SpeedIcon />}
                title="Instant Processing"
                description="Generate bills in seconds by simply scanning your meter QR code, eliminating long wait times and reducing human errors in manual data entry."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard
                icon={<BillIcon />}
                title="Paperless Billing"
                description="Our eco-friendly digital invoicing system reduces paper waste by 90% while keeping all your billing records organized and easily accessible."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard
                icon={<SecurityIcon />}
                title="Secure Transactions"
                description="Bank-grade 256-bit encryption ensures all your billing data and payment information remain completely secure and protected."
              />
            </Grid>
          </Grid>
        </Box>

        {/* How It Works Section */}
        <Paper 
          elevation={0} 
          sx={{ 
            p: { xs: 3, md: 6 }, 
            mb: 15, 
            borderRadius: 3, 
            backgroundColor: alpha(theme.palette.primary.light, 0.05),
            border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          }}
        >
          <Typography 
            variant="h2" 
            component="h2" 
            align="center" 
            gutterBottom 
            sx={{ 
              mb: 8,
              color: theme.palette.primary.dark,
            }}
          >
            How It Works
          </Typography>
          <Grid container spacing={6}>
            {[
              {
                step: '1',
                title: 'Generate Your QR Code',
                description: 'Each customer receives a unique, encrypted QR code permanently linked to their gas account and meter information.',
              },
              {
                step: '2',
                title: 'Scan with Mobile Device',
                description: 'Field agents scan the QR code using our secure mobile application with built-in validation checks.',
              },
              {
                step: '3',
                title: 'Capture Meter Reading',
                description: 'Take a photo of your gas meter or enter the reading manually with automatic validation against previous readings.',
              },
              {
                step: '4',
                title: 'Instant Bill Generation',
                description: 'The system automatically calculates your bill using current rates and sends it digitally to the customer.',
              },
            ].map((item, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Step number={item.step} title={item.title} description={item.description} />
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Action Cards Section */}
        <Box sx={{ mb: 15 }}>
          <Typography 
            variant="h2" 
            component="h2" 
            align="center" 
            gutterBottom 
            sx={{ 
              mb: 8,
              position: 'relative',
              '&::after': {
                content: '""',
                display: 'block',
                width: '80px',
                height: '4px',
                backgroundColor: theme.palette.secondary.main,
                margin: '20px auto 0',
                borderRadius: '2px',
              },
            }}
          >
            Get Started
          </Typography>
          <Grid container spacing={5} sx={{ px: { xs: 0, md: 5 } }}>
            <Grid item xs={12} md={6}>
              <ActionCard
                to="/scan"
                icon={<ScanIcon />}
                title="Scan QR Code"
                description="Scan customer QR codes to quickly access accounts, record meter readings, and generate bills on the spot with our intuitive interface."
                image="https://img.freepik.com/free-vector/smartphone-scanning-qr-code_23-2148624200.jpg"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <ActionCard
                to="/generate"
                icon={<GenerateIcon />}
                title="Generate QR"
                description="Create and manage QR codes for new customer accounts with customizable options for different meter types and locations."
                image="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRjewbaqTCgB3WYaKNiqNWs9VquPmPAvX_jXg&s"
              />
            </Grid>
          </Grid>
        </Box>

        {/* Support Section */}
        <Paper 
          elevation={0} 
          sx={{ 
            p: { xs: 4, md: 6 }, 
            borderRadius: 3, 
            background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
            color: 'white',
            overflow: 'hidden',
            position: 'relative',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: '-50px',
              right: '-50px',
              width: '200px',
              height: '200px',
              borderRadius: '50%',
              background: alpha('#fff', 0.05),
            },
            '&::after': {
              content: '""',
              position: 'absolute',
              bottom: '-80px',
              right: '-80px',
              width: '300px',
              height: '300px',
              borderRadius: '50%',
              background: alpha('#fff', 0.03),
            },
          }}
        >
          <Grid container alignItems="center" spacing={4} position="relative" zIndex={1}>
            <Grid item xs={12} md={8}>
              <Typography variant="h3" component="h3" gutterBottom sx={{ fontWeight: 700 }}>
                Need Help With Our System?
              </Typography>
              <Typography variant="body1" sx={{ mb: 4, opacity: 0.9, maxWidth: '800px' }}>
                Our dedicated customer support team is available 24/7 to assist you with any questions about QR code scanning, bill generation, account management, or technical issues.
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                size="large"
                startIcon={<SupportIcon />}
                sx={{ 
                  px: 5,
                  py: 1.5,
                  fontSize: '1.1rem',
                }}
              >
                Contact Support
              </Button>
            </Grid>
            <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'center' }}>
              <SupportIcon sx={{ fontSize: 150, opacity: 0.1, position: 'relative', zIndex: 0 }} />
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
        <Container maxWidth={false} disableGutters>
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