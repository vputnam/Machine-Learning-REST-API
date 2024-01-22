using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Azure.Storage.Blobs;

namespace MachineLearningFunctions
{
    public class BlobStorageManger
    {

        private String storageConnectionString; 
        private String containerName;

        public BlobStorageManger(String storageConnectionString, String containerName)
        {
            this.storageConnectionString = storageConnectionString;
            this.containerName = containerName;
        }
        
        public async Task UploadONNXBlob(String filename, String filepath)
            {
                BlobServiceClient blobServiceClient = new BlobServiceClient(storageConnectionString);
                
                BlobContainerClient containerClient = blobServiceClient.GetBlobContainerClient(containerName);
                
                BlobClient blobClient = containerClient.GetBlobClient(filename);

                using FileStream uploadFileStream = File.OpenRead(filepath);
                
                await blobClient.UploadAsync(uploadFileStream, true);
                uploadFileStream.Close();
            }
    }
}