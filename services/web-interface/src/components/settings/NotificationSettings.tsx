import React, { useState } from 'react';
import { Bell, Mail, Smartphone, MessageSquare } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { useToast } from '../../hooks/useToast';

export function NotificationSettings() {
  const [settings, setSettings] = useState({
    email: {
      documentProcessing: true,
      systemAlerts: true,
      weeklyReports: false,
      securityNotifications: true,
    },
    push: {
      documentProcessing: false,
      systemAlerts: true,
      queryResults: false,
      mentions: true,
    },
    inApp: {
      documentProcessing: true,
      systemAlerts: true,
      queryResults: true,
      mentions: true,
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleSave = async () => {
    setIsLoading(true);
    try {
      // TODO: Implement API call to save notification settings
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      
      toast({
        title: 'Settings saved',
        description: 'Your notification preferences have been updated.',
        variant: 'success',
      });
    } catch (error: any) {
      toast({
        title: 'Save failed',
        description: error.message || 'Failed to save notification settings',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const updateSetting = (category: keyof typeof settings, setting: string, value: boolean) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [setting]: value,
      },
    }));
  };

  const notificationTypes = [
    {
      id: 'documentProcessing',
      name: 'Document Processing',
      description: 'Notifications when documents finish processing',
    },
    {
      id: 'systemAlerts',
      name: 'System Alerts',
      description: 'Important system notifications and errors',
    },
    {
      id: 'weeklyReports',
      name: 'Weekly Reports',
      description: 'Weekly usage and analytics reports (email only)',
    },
    {
      id: 'securityNotifications',
      name: 'Security Notifications',
      description: 'Login attempts and security-related alerts (email only)',
    },
    {
      id: 'queryResults',
      name: 'Query Results',
      description: 'Notifications for long-running search queries',
    },
    {
      id: 'mentions',
      name: 'Mentions',
      description: 'When you are mentioned in comments or notes',
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Bell className="h-5 w-5" />
          <span>Notification Preferences</span>
        </CardTitle>
        <CardDescription>
          Choose how you want to receive notifications
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Notification Matrix */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">
                    Notification Type
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <div className="flex items-center justify-center space-x-2">
                      <Mail className="h-4 w-4" />
                      <span>Email</span>
                    </div>
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <div className="flex items-center justify-center space-x-2">
                      <Smartphone className="h-4 w-4" />
                      <span>Push</span>
                    </div>
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <div className="flex items-center justify-center space-x-2">
                      <MessageSquare className="h-4 w-4" />
                      <span>In-App</span>
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {notificationTypes.map((type) => (
                  <tr key={type.id} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-4 px-4">
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {type.name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {type.description}
                        </div>
                      </div>
                    </td>
                    <td className="text-center py-4 px-4">
                      {settings.email.hasOwnProperty(type.id) && (
                        <input
                          type="checkbox"
                          checked={settings.email[type.id as keyof typeof settings.email]}
                          onChange={(e) => updateSetting('email', type.id, e.target.checked)}
                          className="rounded border-gray-300 text-primary focus:ring-primary"
                        />
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      {settings.push.hasOwnProperty(type.id) && (
                        <input
                          type="checkbox"
                          checked={settings.push[type.id as keyof typeof settings.push]}
                          onChange={(e) => updateSetting('push', type.id, e.target.checked)}
                          className="rounded border-gray-300 text-primary focus:ring-primary"
                        />
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      {settings.inApp.hasOwnProperty(type.id) && (
                        <input
                          type="checkbox"
                          checked={settings.inApp[type.id as keyof typeof settings.inApp]}
                          onChange={(e) => updateSetting('inApp', type.id, e.target.checked)}
                          className="rounded border-gray-300 text-primary focus:ring-primary"
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Save Button */}
          <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex justify-end">
              <Button
                onClick={handleSave}
                disabled={isLoading}
                className="flex items-center space-x-2"
              >
                <Bell className="h-4 w-4" />
                <span>{isLoading ? 'Saving...' : 'Save Preferences'}</span>
              </Button>
            </div>
          </div>
          
          {/* Additional Info */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h4 className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-2">
              Notification Information
            </h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>• Email notifications are sent to your registered email address</li>
              <li>• Push notifications require browser permission</li>
              <li>• In-app notifications appear in the notification center</li>
              <li>• You can disable all notifications by unchecking all options</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
