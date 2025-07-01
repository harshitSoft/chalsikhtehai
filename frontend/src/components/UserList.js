import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Grid, Card, CardContent, Chip, CircularProgress, Divider, Stack } from '@mui/material';
import axios from 'axios';

const getStatus = (user) => {
    if (user.total_amount === 0) return { label: 'Paid', color: 'success' };
    return { label: 'Unpaid', color: 'error' };
};

const UserSectionTable = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const res = await axios.get('http://localhost:8000/users/user/all');
                setUsers(res.data);
            } catch (err) {
                setError('Failed to fetch users. Make sure the backend is running and accessible at http://localhost:8000/users/user/all. ' + (err.message || ''));
            } finally {
                setLoading(false);
            }
        };
        fetchUsers();
    }, []);

    return (
        <Box className="user-section-table" sx={{ maxWidth: 'lg', mx: 'auto', mt: 4, p: 2 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 3 }}>
                    All Users
                </Typography>
                {loading ? (
                    <Box sx={{ textAlign: 'center', py: 4 }}>
                        <CircularProgress />
                    </Box>
                ) : error ? (
                    <Typography color="error">{error}</Typography>
                ) : users.length === 0 ? (
                    <Typography>No users found.</Typography>
                ) : (
                    <Grid container spacing={3}>
                        {users.map((user) => {
                            const status = getStatus(user);
                            const isLate = user.late_fees > 0;
                            return (
                                <Grid item xs={12} sm={6} md={4} key={user.id}>
                                    <Card sx={{ borderRadius: 3, boxShadow: 4, height: '100%' }}>
                                        <CardContent>
                                            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                                    {user.username}
                                                </Typography>
                                                <Chip label={status.label} color={status.color} size="small" />
                                            </Stack>
                                            <Divider sx={{ mb: 1 }} />
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                <b>Email:</b> {user.email}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                <b>Zone:</b> {user.zone || '-'}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                <b>Meter Number:</b> {user.meter_number || '-'}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                <b>Contact:</b> {user.contact_number || '-'}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                <b>Address:</b> {user.address || '-'}
                                            </Typography>
                                            <Divider sx={{ my: 1 }} />
                                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                                <Typography variant="body2"><b>Total Amount:</b></Typography>
                                                <Chip label={`₹${user.total_amount?.toFixed(2) || '0.00'}`} color={status.color} size="small" />
                                            </Stack>
                                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                                <Typography variant="body2"><b>Late Fees:</b></Typography>
                                                <Chip label={`₹${user.late_fees?.toFixed(2) || '0.00'}`} color={isLate ? 'error' : 'success'} size="small" />
                                            </Stack>
                                            <Typography variant="body2" color="text.secondary">
                                                <b>Last Reading Date:</b> {user.last_reading_date ? new Date(user.last_reading_date).toLocaleDateString() : '-'}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            );
                        })}
                    </Grid>
                )}
            </Paper>
        </Box>
    );
};

export default UserSectionTable; 