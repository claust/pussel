param name string
param location string
param containerRegistryName string
param skuName string
param storageAccountName string
param storageAccountId string
param prefix string

var appServicePlanName = '${name}-plan'

resource appServicePlan 'Microsoft.Web/serverfarms@2021-02-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: skuName
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

resource appService 'Microsoft.Web/sites@2021-02-01' = {
  name: name
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|mcr.microsoft.com/appsvc/staticsite:latest'
      appCommandLine: 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
      appSettings: [
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: 'true'
        }
        {
          name: 'UPLOAD_DIR'
          value: '/tmp/uploads'
        }
        {
          name: 'AZURE_STORAGE_CONNECTION_STRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccountName};AccountKey=${listKeys(storageAccountId, '2021-04-01').keys[0].value};EndpointSuffix=core.windows.net'
        }
        {
          name: 'USE_AZURE_STORAGE'
          value: 'true'
        }
        {
          name: 'BACKEND_CORS_ORIGINS'
          value: '["https://pussel-frontend.azurewebsites.net", "http://localhost:3000", "https://pussel.thomasen.dk"]'
        }
        {
          name: 'PUBLIC_URL'
          value: 'https://pussel.thomasen.dk'
        }
        {
          name: 'WEBSITES_ENABLE_APP_SERVICE_STORAGE'
          value: 'false'
        }
      ]
    }
  }
}

// Try to reference the existing certificate - will succeed if exists
resource existingCertificate 'Microsoft.Web/certificates@2021-02-01' existing = {
  name: 'pussel.thomasen.dk-pussel-backend-dev'
  scope: resourceGroup()
}

// Domain binding using the existing certificate
resource customDomainBinding 'Microsoft.Web/sites/hostNameBindings@2021-02-01' = {
  parent: appService
  name: 'pussel.thomasen.dk'
  properties: {
    hostNameType: 'Verified'
    sslState: 'SniEnabled'
    thumbprint: existingCertificate.properties.thumbprint
    customHostNameDnsRecordType: 'CName'
  }
}

// Update CORS to include the custom domain
resource corsSettings 'Microsoft.Web/sites/config@2021-02-01' = {
  parent: appService
  name: 'web'
  properties: {
    cors: {
      allowedOrigins: [
        'https://pussel-frontend.azurewebsites.net'
        'http://localhost:3000'
        'https://pussel.thomasen.dk'
      ]
    }
  }
}

output name string = appService.name
output url string = 'https://${appService.properties.defaultHostName}'
