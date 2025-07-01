import React, { useState } from 'react';
import { Box, Typography, Paper, TextField, Button, MenuItem, Divider, List, ListItem, ListItemText, Select, InputLabel, FormControl, Alert } from '@mui/material';
import axios from 'axios';

const roles = [
  { value: 'admin', label: 'Admin' },
  { value: 'staff', label: 'Staff' },
  { value: 'customer', label: 'Customer' },
];

const AdminPanel = () => {
  // Admin login state
  const [adminLoginId, setAdminLoginId] = useState('');
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
  const [message, setMessage] = useState('');

  // Admin login handler
  const handleAdminLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await axios.get('http://localhost:8000/users/user/all');
      const admin = res.data.find(u => u.id === parseInt(adminLoginId) && u.role === 'admin');
      if (admin) {
        setIsAdmin(true);
        setAdminName(admin.username);
      } else {
        setLoginError('Invalid Admin ID');
      }
    } catch {
      setLoginError('Failed to verify admin');
    }
  };

  // Create user (admin, staff, customer)
  const handleCreate = async (e) => {
    e.preventDefault();
    setMessage('');
    let url = '';
    if (!form.username || !form.email || !form.role || (form.role !== 'admin' && !form.created_by)) {
      setMessage('Please fill all required fields.');
      return;
    }
    if (form.role === 'admin') url = 'http://localhost:8000/users/user/create-admin';
    else if (form.role === 'staff') url = 'http://localhost:8000/users/user/create-staff';
    else url = 'http://localhost:8000/users/user/create-customer';
    // Only allow staff creation if logged in as admin
    if (form.role === 'staff' && !isAdmin) {
      setMessage('Only admins can create staff.');
      return;
    }
    try {
      await axios.post(url, form);
      setMessage('User created!');
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  // List all admins
  const fetchAdmins = async () => {
    setMessage('');
    try {
      const res = await axios.get('http://localhost:8000/users/user/all-admins');
      setAdmins(res.data);
    } catch (err) {
      setMessage('Error fetching admins: ' + (err.response?.data?.detail || err.message));
    }
  };

  // List staff for an admin
  const fetchStaff = async () => {
    setMessage('');
    if (!adminId) {
      setMessage('Please enter Admin ID.');
      return;
    }
    try {
      const res = await axios.get(`http://localhost:8000/users/user/admin/${adminId}/staff`);
      setStaff(res.data);
    } catch (err) {
      setMessage('Error fetching staff: ' + (err.response?.data?.detail || err.message));
    }
  };

  // List customers for a staff
  const fetchCustomers = async () => {
    setMessage('');
    if (!staffId) {
      setMessage('Please enter Staff ID.');
      return;
    }
    try {
      const res = await axios.get(`http://localhost:8000/users/user/staff/${staffId}/customers`);
      setCustomers(res.data);
    } catch (err) {
      setMessage('Error fetching customers: ' + (err.response?.data?.detail || err.message));
    }
  };

  // If not logged in as admin, show login form
  if (!isAdmin) {
    return (
      <Box sx={{ maxWidth: 400, mx: 'auto', mt: 8, p: 3, borderRadius: 2, boxShadow: 3, background: '#fff' }}>
        <Typography variant="h5" sx={{ mb: 2 }}>Admin Login</Typography>
        <form onSubmit={handleAdminLogin}>
          <TextField
            label="Admin ID"
            value={adminLoginId}
            onChange={e => setAdminLoginId(e.target.value)}
            type="number"
            fullWidth
            required
            sx={{ mb: 2 }}
          />
          <Button type="submit" variant="contained" color="primary" fullWidth>Login</Button>
        </form>
        {loginError && <Alert severity="error" sx={{ mt: 2 }}>{loginError}</Alert>}
      </Box>
    );
  }

  // Admin panel UI
  return (
    <Box sx={{ maxWidth: 700, mx: 'auto', mt: 4, p: 2 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 4, background: '#1976d2', color: '#fff' }}>
        <Typography variant="h5">Welcome, Admin {adminName} (ID: {adminLoginId})</Typography>
      </Paper>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          Create User (Admin/Staff/Customer)
        </Typography>
        <form onSubmit={handleCreate}>
          <TextField
            fullWidth
            label="Username"
            name="username"
            value={form.username}
            onChange={e => setForm({ ...form, username: e.target.value })}
            margin="normal"
            required
          />
          <TextField
            fullWidth
            label="Email"
            name="email"
            value={form.email}
            onChange={e => setForm({ ...form, email: e.target.value })}
            margin="normal"
            required
            type="email"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Role</InputLabel>
            <Select
              value={form.role}
              label="Role"
              onChange={e => setForm({ ...form, role: e.target.value })}
            >
              {roles.map(r => (
                <MenuItem key={r.value} value={r.value}>{r.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          {form.role !== 'admin' && (
            <TextField
              fullWidth
              label="Created By (User ID)"
              name="created_by"
              value={form.created_by}
              onChange={e => setForm({ ...form, created_by: e.target.value })}
              margin="normal"
              required
            />
          )}
          <TextField
            fullWidth
            label="Area"
            name="area"
            value={form.area}
            onChange={e => setForm({ ...form, area: e.target.value })}
            margin="normal"
          />
          <Button type="submit" variant="contained" color="primary" sx={{ mt: 2 }}>
            Create
          </Button>
        </form>
        {message && <Typography sx={{ mt: 2 }}>{message}</Typography>}
      </Paper>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6">List All Admins</Typography>
        <Button onClick={fetchAdmins} variant="outlined" sx={{ my: 2 }}>Fetch Admins</Button>
        <List>
          {admins.map(a => (
            <ListItem key={a.id} divider>
              <ListItemText primary={`${a.username} (${a.email})`} secondary={`Area: ${a.area}, ID: ${a.id}`} />
            </ListItem>
          ))}
        </List>
      </Paper>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6">List Staff for Admin</Typography>
        <TextField
          fullWidth
          label="Admin ID"
          value={adminId}
          onChange={e => setAdminId(e.target.value)}
          margin="normal"
        />
        <Button onClick={fetchStaff} variant="outlined" sx={{ my: 2 }}>Fetch Staff</Button>
        <List>
          {staff.map(s => (
            <ListItem key={s.id} divider>
              <ListItemText primary={`${s.username} (${s.email})`} secondary={`Area: ${s.area}, ID: ${s.id}`} />
            </ListItem>
          ))}
        </List>
      </Paper>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h6">List Customers for Staff</Typography>
        <TextField
          fullWidth
          label="Staff ID"
          value={staffId}
          onChange={e => setStaffId(e.target.value)}
          margin="normal"
        />
        <Button onClick={fetchCustomers} variant="outlined" sx={{ my: 2 }}>Fetch Customers</Button>
        <List>
          {customers.map(c => (
            <ListItem key={c.id} divider>
              <ListItemText primary={`${c.username} (${c.email})`} secondary={`Area: ${c.area}, ID: ${c.id}`} />
            </ListItem>
          ))}
        </List>
      </Paper>
    </Box>
  );
};

export default AdminPanel; 