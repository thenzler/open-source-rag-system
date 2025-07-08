import React, { useState } from 'react';
import {
  Settings,
  User,
  Database,
  Shield,
  Bell,
  Palette,
  Key,
  Download,
  Upload,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { UserSettings } from '../components/settings/UserSettings';
import { SystemSettings } from '../components/settings/SystemSettings';
import { SecuritySettings } from '../components/settings/SecuritySettings';
import { NotificationSettings } from '../components/settings/NotificationSettings';
import { useAppStore } from '../store/useAppStore';

type SettingsTab = 'profile' | 'system' | 'security' | 'notifications' | 'appearance' | 'api';

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('profile');
  const { user, theme, setTheme } = useAppStore();

  const tabs = [
    {
      id: 'profile' as const,
      label: 'Profile',
      icon: User,
      description: 'Manage your account information',
    },
    {
      id: 'system' as const,
      label: 'System',
      icon: Database,
      description: 'Configure system settings',
    },
    {
      id: 'security' as const,
      label: 'Security',
      icon: Shield,
      description: 'Security and privacy settings',
    },
    {
      id: 'notifications' as const,
      label: 'Notifications',
      icon: Bell,
      description: 'Notification preferences',
    },
    {
      id: 'appearance' as const,
      label: 'Appearance',
      icon: Palette,
      description: 'Theme and display settings',
    },
    {
      id: 'api' as const,
      label: 'API Keys',
      icon: Key,
      description: 'Manage API access keys',
    },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return <UserSettings />;
      case 'system':
        return <SystemSettings />;
      case 'security':
        return <SecuritySettings />;
      case 'notifications':
        return <NotificationSettings />;
      case 'appearance':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Appearance Settings</CardTitle>
              <CardDescription>
                Customize the look and feel of the application
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block">
                  Theme
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    onClick={() => setTheme('light')}
                    className={`p-4 border rounded-lg text-center transition-all ${
                      theme === 'light'
                        ? 'border-primary bg-primary/5 text-primary'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <div className="w-8 h-6 bg-white border border-gray-300 rounded mx-auto mb-2" />
                    <span className="text-sm font-medium">Light</span>
                  </button>
                  <button
                    onClick={() => setTheme('dark')}
                    className={`p-4 border rounded-lg text-center transition-all ${
                      theme === 'dark'
                        ? 'border-primary bg-primary/5 text-primary'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <div className="w-8 h-6 bg-gray-800 border border-gray-600 rounded mx-auto mb-2" />
                    <span className="text-sm font-medium">Dark</span>
                  </button>
                  <button
                    onClick={() => setTheme('system')}
                    className={`p-4 border rounded-lg text-center transition-all ${
                      theme === 'system'
                        ? 'border-primary bg-primary/5 text-primary'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <div className="w-8 h-6 bg-gradient-to-r from-white to-gray-800 border border-gray-300 rounded mx-auto mb-2" />
                    <span className="text-sm font-medium">System</span>
                  </button>
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block">
                  Font Size
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <button className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg text-center hover:border-gray-300 transition-colors">
                    <span className="text-xs">Small</span>
                  </button>
                  <button className="p-3 border border-primary bg-primary/5 text-primary rounded-lg text-center">
                    <span className="text-sm">Medium</span>
                  </button>
                  <button className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg text-center hover:border-gray-300 transition-colors">
                    <span className="text-base">Large</span>
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      case 'api':
        return (
          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
              <CardDescription>
                Manage your API access keys for programmatic access
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                    Personal Access Token
                  </h3>
                  <p className="text-sm text-gray-500 mt-1">
                    Use this token to authenticate API requests
                  </p>
                </div>
                <Button size="sm">
                  Generate New Key
                </Button>
              </div>
              
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      Default API Key
                    </span>
                    <span className="text-xs text-gray-500 bg-white dark:bg-gray-700 px-2 py-1 rounded">
                      Created 2 days ago
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <code className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                      rag_sk_••••••••••••••••••••••••••••••••••••
                    </code>
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        Copy
                      </Button>
                      <Button variant="destructive" size="sm">
                        Revoke
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-2">
                  API Usage Guidelines
                </h4>
                <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                  <li>• Rate limit: 1000 requests per hour</li>
                  <li>• Keep your API keys secure and never share them</li>
                  <li>• Rotate keys regularly for security</li>
                  <li>• Use HTTPS for all API requests</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Manage your account, system preferences, and application settings
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <nav className="space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 text-left rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary text-primary-foreground'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <div>
                    <span className="font-medium">{tab.label}</span>
                    <p className="text-xs opacity-70 hidden sm:block">
                      {tab.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
}
