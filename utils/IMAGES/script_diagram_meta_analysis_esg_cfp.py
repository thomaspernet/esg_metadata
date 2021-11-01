
from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS
from diagrams.aws.database import Redshift, RDS
from diagrams.aws.integration import SQS
from diagrams.aws.storage import S3

with Diagram("META ANALYSIS ESG CFP", show=False, filename="/home/ec2-user/esg_metadata/utils/IMAGES/meta_analysis_esg_cfp", outformat="jpg"):

     temp_1 = S3('papers_meta_analysis_new')

     with Cluster("FINAL"):

         temp_final_0 = Redshift('meta_analysis_esg_cfp')


     temp_final_0 << temp_1
