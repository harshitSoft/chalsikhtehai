import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  TextField, 
  Button, 
  MenuItem, 
  Divider, 
  List, 
  ListItem, 
  ListItemText, 
  Select, 
  InputLabel, 
  FormControl, 
  Alert,
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  CircularProgress,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Avatar,
  Chip,
  LinearProgress,
  IconButton,
  Collapse
} from '@mui/material';
import { 
  Lock as LockIcon, 
  PersonAdd as PersonAddIcon,
  People as PeopleIcon,
  Business as BusinessIcon,
  Person as PersonIcon,
  Receipt as ReceiptIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Search as SearchIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import axios from 'axios';
import UserSectionTable from './UserList';
import { teal, indigo, deepPurple, blue, pink } from '@mui/material/colors';

const roles = [
  { value: 'admin', label: 'Admin', icon: <BusinessIcon />, color: indigo[500] },
  { value: 'staff', label: 'Staff', icon: <PeopleIcon />, color: teal[500] },
  { value: 'customer', label: 'Customer', icon: <PersonIcon />, color: blue[500] },
];

const ExpandableSection = styled(Paper)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  overflow: 'hidden',
  transition: 'box-shadow 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[6],
  },
}));

const SectionHeader = styled(CardHeader)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.common.white,
  cursor: 'pointer',
  '& .MuiCardHeader-action': {
    margin: 0,
    alignSelf: 'center',
  }
}));

const RoleChip = styled(Chip)(({ theme, role }) => ({
  backgroundColor: roles.find(r => r.value === role)?.color || theme.palette.grey[500],
  color: theme.palette.common.white,
  fontWeight: 500,
  marginLeft: theme.spacing(1),
}));

const AdminPanel = () => {
  // Admin login state
  const [adminLoginCreatedBy, setAdminLoginCreatedBy] = useState('');
  const [adminName, setAdminName] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [loginError, setLoginError] = useState('');

  // User creation and listing state
  const [form, setForm] = useState({
    username: '',
    email: '',
    role: 'admin',
    created_by: '',
    area: '',
  });
  const [admins, setAdmins] = useState([]);
  const [staff, setStaff] = useState([]);
  const [customers, setCustomers] = useState([]);
  const [adminId, setAdminId] = useState('');
  const [staffId, setStaffId] = useState('');
  const [message, setMessage] = useState({ text: '', severity: 'info' });
  const [allStaff, setAllStaff] = useState([]);
  const [loadingStaff, setLoadingStaff] = useState(false);
  const [staffError, setStaffError] = useState('');
  
  // Expanded sections state
  const [expanded, setExpanded] = useState({
    createUser: true,
    allAdmins: false,
    adminStaff: false,
    staffCustomers: false,
    staffBills: true,
    allUsers: false
  });

  // Meter number search state
  const [meterNumberSearch, setMeterNumberSearch] = useState('');
  const [meterUserResult, setMeterUserResult] = useState(null);

  const toggleSection = (section) => {
    setExpanded(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // Admin login handler
  const handleAdminLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await axios.get('http://localhost:8000/users/user/all');
      const admin = res.data.find(u => String(u.created_by) === adminLoginCreatedBy && u.role === 'admin');
      if (admin) {
        setIsAdmin(true);
        setAdminName(admin.username);
      } else {
        setLoginError('Invalid Created By value for Admin');
      }
    } catch {
      setLoginError('Failed to verify admin');
    }
  };

  // Create user (admin, staff, customer)
  const handleCreate = async (e) => {
    e.preventDefault();
    setMessage({ text: '', severity: 'info' });
    
    if (!form.username || !form.email || !form.role || !form.created_by) {
      setMessage({ text: 'Please fill all required fields.', severity: 'error' });
      return;
    }
    
    let url = '';
    if (form.role === 'admin') url = 'http://localhost:8000/users/user/create-admin';
    else if (form.role === 'staff') url = 'http://localhost:8000/users/user/create-staff';
    else url = 'http://localhost:8000/users/user/create-customer';
    
    if (form.role === 'staff' && !isAdmin) {
      setMessage({ text: 'Only admins can create staff.', severity: 'error' });
      return;
    }
    
    try {
      await axios.post(url, form);
      setMessage({ text: 'User created successfully!', severity: 'success' });
      setForm(prev => ({ ...prev, username: '', email: '', created_by: '', area: '' }));
      if (form.role === 'admin') fetchAdmins();
    } catch (err) {
      setMessage({ text: 'Error: ' + (err.response?.data?.detail || err.message), severity: 'error' });
    }
  };

  // List all admins
  const fetchAdmins = async () => {
    setMessage({ text: '', severity: 'info' });
    try {
      const res = await axios.get('http://localhost:8000/users/user/all-admins');
      setAdmins(res.data);
    } catch (err) {
      setMessage({ text: 'Error fetching admins: ' + (err.response?.data?.detail || err.message), severity: 'error' });
    }
  };

  // List staff for an admin
  const fetchStaff = async () => {
    setMessage({ text: '', severity: 'info' });
    if (!adminId) {
      setMessage({ text: 'Please enter Admin ID.', severity: 'error' });
      return;
    }
    try {
      const res = await axios.get(`http://localhost:8000/users/user/admin/${adminId}/staff`);
      setStaff(res.data);
    } catch (err) {
      setMessage({ text: 'Error fetching staff: ' + (err.response?.data?.detail || err.message), severity: 'error' });
    }
  };

  // List customers for a staff
  const fetchCustomers = async () => {
    setMessage({ text: '', severity: 'info' });
    if (!staffId) {
      setMessage({ text: 'Please enter Staff ID.', severity: 'error' });
      return;
    }
    try {
      const res = await axios.get(`http://localhost:8000/users/user/staff/${staffId}/customers`);
      setCustomers(res.data);
    } catch (err) {
      setMessage({ text: 'Error fetching customers: ' + (err.response?.data?.detail || err.message), severity: 'error' });
    }
  };

  // Fetch all staff for bill_count section
  const fetchAllStaff = async () => {
    setLoadingStaff(true);
    setStaffError('');
    try {
      const res = await axios.get('http://localhost:8000/users/user/all');
      setAllStaff(res.data.filter(u => u.role === 'staff'));
    } catch (err) {
      setStaffError('Failed to fetch staff.');
    } finally {
      setLoadingStaff(false);
    }
  };

  // Fetch staff on admin login
  useEffect(() => {
    if (isAdmin) fetchAllStaff();
  }, [isAdmin]);

  // Handle meter number search
  const handleMeterNumberSearch = async () => {
    setMeterUserResult(null);
    if (!meterNumberSearch) return;
    try {
      const res = await axios.get(`http://localhost:8000/users/user/all`);
      const user = res.data.find(u => u.meter_number === meterNumberSearch);
      if (user) setMeterUserResult(user);
      else setMeterUserResult(null);
    } catch (err) {
      setMeterUserResult(null);
    }
  };

  // If not logged in as admin, show login form
  if (!isAdmin) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          p: 2
        }}
      >
        <Card sx={{ maxWidth: 450, width: '100%', boxShadow: 3 }}>
          <CardHeader
            avatar={
              <Avatar sx={{ bgcolor: deepPurple[500] }}>
                <LockIcon />
              </Avatar>
            }
            title={
              <Typography variant="h5" component="div">
                Admin Portal
              </Typography>
            }
            subheader="Enter your superadmin credentials"
            sx={{ bgcolor: deepPurple[700], color: 'white', textAlign: 'center' }}
          />
          <CardContent>
            <form onSubmit={handleAdminLogin}>
              <TextField
                label="Enter Admin Code"
                value={adminLoginCreatedBy}
                onChange={e => setAdminLoginCreatedBy(e.target.value)}
                type="text"
                fullWidth
                required
                margin="normal"
                variant="outlined"
              />
              <Button
                type="submit"
                variant="contained"
                color="primary"
                fullWidth
                size="large"
                sx={{ mt: 2, py: 1.5 }}
                startIcon={<LockIcon />}
              >
                Login
              </Button>
            </form>
            {loginError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {loginError}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Box>
    );
  }

  // Admin panel UI
  return (
    <Box sx={{ p: { xs: 2, md: 3 }, bgcolor: 'background.default', minHeight: '100vh' }}>
      <Grid container spacing={3}>
        {/* Header */}
        <Grid item xs={12}>
          <Card sx={{ bgcolor: 'primary.main', color: 'common.white' }}>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'common.white', color: 'primary.main', mr: 2 }}>
                  <BusinessIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="h1">
                    Admin Dashboard
                  </Typography>
                  <Typography variant="subtitle1">
                    Welcome back, {adminName}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Create User Section */}
        <Grid item xs={12} md={6}>
          <ExpandableSection>
            <SectionHeader
              title="Create New "
              subheader="Add admin, staff or customer"
              avatar={
                <Avatar sx={{ bgcolor: pink[500] }}>
                  <PersonAddIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('createUser')}>
                  {expanded.createUser ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('createUser')}
            />
            <Collapse in={expanded.createUser}>
              <CardContent>
                <form onSubmit={handleCreate}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Username"
                        name="username"
                        value={form.username}
                        onChange={e => setForm({ ...form, username: e.target.value })}
                        margin="normal"
                        required
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Email"
                        name="email"
                        value={form.email}
                        onChange={e => setForm({ ...form, email: e.target.value })}
                        margin="normal"
                        required
                        type="email"
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Role</InputLabel>
                        <Select
                          value={form.role}
                          label="Role"
                          onChange={e => setForm({ ...form, role: e.target.value })}
                          variant="outlined"
                        >
                          {roles.map(r => (
                            <MenuItem key={r.value} value={r.value}>
                              <Box display="flex" alignItems="center">
                                <Box mr={1}>{r.icon}</Box>
                                {r.label}
                              </Box>
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Unique Code"
                        name="created_by"
                        value={form.created_by}
                        onChange={e => setForm({ ...form, created_by: e.target.value })}
                        margin="normal"
                        required
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Area"
                        name="area"
                        value={form.area}
                        onChange={e => setForm({ ...form, area: e.target.value })}
                        margin="normal"
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        fullWidth
                        size="large"
                        startIcon={<PersonAddIcon />}
                      >
                        Create User
                      </Button>
                    </Grid>
                  </Grid>
                </form>
                {message.text && (
                  <Alert severity={message.severity} sx={{ mt: 2 }}>
                    {message.text}
                  </Alert>
                )}
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>

        {/* All Admins Section */}
        <Grid item xs={12} md={6}>
          <ExpandableSection>
            <SectionHeader
              title="Search User by Meter Number"
              subheader="Find user details by meter number"
              avatar={
                <Avatar sx={{ bgcolor: indigo[500] }}>
                  <BusinessIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('allAdmins')}>
                  {expanded.allAdmins ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('allAdmins')}
            />
            <Collapse in={expanded.allAdmins}>
              <CardContent>
                <TextField
                  fullWidth
                  label="Meter Number"
                  value={meterNumberSearch || ''}
                  onChange={e => setMeterNumberSearch(e.target.value)}
                  margin="normal"
                  variant="outlined"
                />
                <Button
                  onClick={handleMeterNumberSearch}
                  variant="outlined"
                  color="primary"
                  fullWidth
                  sx={{ mb: 2 }}
                  startIcon={<SearchIcon />}
                >
                  Search User
                </Button>
                {meterUserResult ? (
                  <List dense>
                    <ListItem divider>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center">
                            {meterUserResult.username}
                            <RoleChip label={meterUserResult.role} role={meterUserResult.role} size="small" />
                          </Box>
                        }
                        secondary={
                          <>
                            <Box component="span" display="block">Email: {meterUserResult.email}</Box>
                            <Box component="span" display="block">ID: {meterUserResult.id} • Area: {meterUserResult.area || 'N/A'}</Box>
                          </>
                        }
                      />
                    </ListItem>
                  </List>
                ) : (
                  <Typography variant="body2" color="textSecondary" align="center" sx={{ py: 2 }}>
                    Enter a meter number and click search to find user.
                  </Typography>
                )}
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>

        {/* Admin Staff Section */}
        <Grid item xs={12} md={6}>
          <ExpandableSection>
            <SectionHeader
              title="Staff Members"
              subheader="Staff managed by specific admin"
              avatar={
                <Avatar sx={{ bgcolor: teal[500] }}>
                  <PeopleIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('adminStaff')}>
                  {expanded.adminStaff ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('adminStaff')}
            />
            <Collapse in={expanded.adminStaff}>
              <CardContent>
                <TextField
                  fullWidth
                  label="Staff Code"
                  value={adminId}
                  onChange={e => setAdminId(e.target.value)}
                  margin="normal"
                  variant="outlined"
                />
                <Button
                  onClick={fetchStaff}
                  variant="outlined"
                  color="primary"
                  fullWidth
                  sx={{ mb: 2 }}
                  startIcon={<PeopleIcon />}
                >
                  Fetch Staff
                </Button>
                {staff.length > 0 ? (
                  <List dense>
                    {staff.map(s => (
                      <ListItem key={s.id} divider>
                        <ListItemText
                          primary={
                            <Box display="flex" alignItems="center">
                              {s.username}
                              <RoleChip label="Staff" role="staff" size="small" />
                            </Box>
                          }
                          secondary={
                            <>
                              <Box component="span" display="block">Email: {s.email}</Box>
                              <Box component="span" display="block">ID: {s.id} • Area: {s.area || 'N/A'}</Box>
                            </>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Typography variant="body2" color="textSecondary" align="center" sx={{ py: 2 }}>
                    Enter Staff Code and click the button to fetch staff members.
                  </Typography>
                )}
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>

        {/* Staff Customers Section */}
        <Grid item xs={12} md={6}>
          <ExpandableSection>
            <SectionHeader
              title="Unpaid Customers"
              subheader="Customers managed by Unpaid status"
              avatar={
                <Avatar sx={{ bgcolor: blue[500] }}>
                  <PersonIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('staffCustomers')}>
                  {expanded.staffCustomers ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('staffCustomers')}
            />
            <Collapse in={expanded.staffCustomers}>
              <CardContent>
                <TextField
                  fullWidth
                  label="Staff ID"
                  value={staffId}
                  onChange={e => setStaffId(e.target.value)}
                  margin="normal"
                  variant="outlined"
                />
                <Button
                  onClick={fetchCustomers}
                  variant="outlined"
                  color="primary"
                  fullWidth
                  sx={{ mb: 2 }}
                  startIcon={<PersonIcon />}
                >
                  Fetch Customers
                </Button>
                {customers.length > 0 ? (
                  <List dense>
                    {customers.map(c => (
                      <ListItem key={c.id} divider>
                        <ListItemText
                          primary={
                            <Box display="flex" alignItems="center">
                              {c.username}
                              <RoleChip label="Customer" role="customer" size="small" />
                            </Box>
                          }
                          secondary={
                            <>
                              <Box component="span" display="block">Email: {c.email}</Box>
                              <Box component="span" display="block">ID: {c.id} • Area: {c.area || 'N/A'}</Box>
                            </>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Typography variant="body2" color="textSecondary" align="center" sx={{ py: 2 }}>
                    Enter Staff ID and click the button to fetch customers.
                  </Typography>
                )}
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>

        {/* Staff Bill Count Section */}
        <Grid item xs={12}>
          <ExpandableSection>
            <SectionHeader
              title="Staff Performance"
              subheader="Bill count overview by staff members"
              avatar={
                <Avatar sx={{ bgcolor: deepPurple[500] }}>
                  <ReceiptIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('staffBills')}>
                  {expanded.staffBills ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('staffBills')}
            />
            <Collapse in={expanded.staffBills}>
              <CardContent>
                {loadingStaff ? (
                  <Box sx={{ textAlign: 'center', py: 2 }}>
                    <CircularProgress />
                    <Typography variant="body2" sx={{ mt: 1 }}>Loading staff data...</Typography>
                  </Box>
                ) : staffError ? (
                  <Alert severity="error">{staffError}</Alert>
                ) : allStaff.length === 0 ? (
                  <Typography variant="body2" color="textSecondary" align="center" sx={{ py: 2 }}>
                    No staff members found.
                  </Typography>
                ) : (
                  <TableContainer component={Paper} variant="outlined">
                    <Table>
                      <TableHead>
                        <TableRow sx={{ bgcolor: 'action.hover' }}>
                          <TableCell><b>Staff Member</b></TableCell>
                          <TableCell><b>Contact</b></TableCell>
                          <TableCell><b>Area</b></TableCell>
                          <TableCell align="right"><b>Bill Count</b></TableCell>
                          <TableCell width="30%"><b>Performance</b></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {allStaff.map(staff => {
                          const billCount = staff.bill_count || 0;
                          const progress = Math.min(billCount * 10, 100); // Simple scaling for visualization
                          return (
                            <TableRow key={staff.id} hover>
                              <TableCell>
                                <Box display="flex" alignItems="center">
                                  <Avatar sx={{ bgcolor: teal[100], color: teal[800], mr: 2, width: 32, height: 32 }}>
                                    {staff.username.charAt(0).toUpperCase()}
                                  </Avatar>
                                  {staff.username}
                                </Box>
                              </TableCell>
                              <TableCell>{staff.email}</TableCell>
                              <TableCell>{staff.area || '-'}</TableCell>
                              <TableCell align="right">
                                <Chip 
                                  label={billCount} 
                                  color={billCount > 0 ? 'primary' : 'default'}
                                  variant={billCount > 0 ? 'filled' : 'outlined'}
                                />
                              </TableCell>
                              <TableCell>
                                <Box display="flex" alignItems="center">
                                  <Box width="100%" mr={1}>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={progress} 
                                      color={
                                        billCount > 15 ? 'success' : 
                                        billCount > 5 ? 'primary' : 
                                        'secondary'
                                      }
                                    />
                                  </Box>
                                  <Typography variant="body2" color="textSecondary">
                                    {progress}%
                                  </Typography>
                                </Box>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>

        {/* All Users Section */}
        <Grid item xs={12}>
          <ExpandableSection>
            <SectionHeader
              title="Complete User Directory"
              subheader="Detailed view of all system users"
              avatar={
                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                  <PeopleIcon />
                </Avatar>
              }
              action={
                <IconButton aria-label="expand" onClick={() => toggleSection('allUsers')}>
                  {expanded.allUsers ? <ExpandLessIcon sx={{ color: 'white' }} /> : <ExpandMoreIcon sx={{ color: 'white' }} />}
                </IconButton>
              }
              onClick={() => toggleSection('allUsers')}
            />
            <Collapse in={expanded.allUsers}>
              <CardContent>
                <UserSectionTable />
              </CardContent>
            </Collapse>
          </ExpandableSection>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdminPanel;