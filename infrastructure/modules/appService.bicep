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
      linuxFxVersion: 'DOCKER|${containerRegistryName}.azurecr.io/pussel-backend:latest'
      appCommandLine: 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
      appSettings: [
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${containerRegistryName}.azurecr.io'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_USERNAME'
          value: containerRegistryName
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_PASSWORD'
          value: '@Microsoft.KeyVault(SecretUri=kvReference)'
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
          value: '["https://pussel-frontend.azurewebsites.net", "http://localhost:3000"]'
        }
      ]
    }
  }
}

// Create managed certificate
resource managedCertificate 'Microsoft.Web/certificates@2021-02-01' = {
  name: '${prefix}-cert-pussel'
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    canonicalName: 'pussel.thomasen.dk'
  }
}

// Bind the certificate to the custom domain
resource sslBinding 'Microsoft.Web/sites/hostNameBindings@2021-02-01' = {
  parent: appService
  name: 'pussel.thomasen.dk'
  properties: {
    siteName: appService.name
    hostNameType: 'Verified'
    sslState: 'SniEnabled'
    thumbprint: managedCertificate.properties.thumbprint
    customHostNameDnsRecordType: 'CName'
  }
}

output name string = appService.name
output url string = 'https://${appService.properties.defaultHostName}'
