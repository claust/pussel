@description('Environment name')
param environmentName string = 'dev'

@description('Azure region')
param location string = resourceGroup().location

@description('App service plan SKU')
param appServicePlanSku string = 'B1'

// Resource naming
var prefix = 'ski-${environmentName}-clth-pussel'
var acr_name = replace('${prefix}-acr', '-', '')
var storage_name = replace('${prefix}-st', '-', '')
var app_name = '${prefix}-app'

// Container Registry
module acr './modules/containerRegistry.bicep' = {
  name: 'acr-deployment'
  params: {
    name: acr_name
    location: location
  }
}

// Storage Account
module storage './modules/storageAccount.bicep' = {
  name: 'storage-deployment'
  params: {
    name: storage_name
    location: location
  }
}

// App Service
module appService './modules/appService.bicep' = {
  name: 'appservice-deployment'
  params: {
    name: app_name
    location: location
    skuName: appServicePlanSku
    storageAccountName: storage.outputs.name
    storageAccountId: storage.outputs.storageId
  }
}

// Outputs
output appServiceName string = appService.outputs.name
output appServiceUrl string = appService.outputs.url
output containerRegistryName string = acr.outputs.name
output storageAccountName string = storage.outputs.name
