param name string
param location string
param containerRegistryName string
param skuName string
param storageAccountName string
param storageAccountKey string

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
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccountName};AccountKey=${storageAccountKey};EndpointSuffix=core.windows.net'
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

output name string = appService.name
output url string = 'https://${appService.properties.defaultHostName}'
