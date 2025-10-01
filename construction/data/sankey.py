import random
import pandas as pd
import numpy as np
import os

# Output directory
output_dir = "csv/sankey/"
os.makedirs(output_dir, exist_ok=True)

"""             Begin Sankey Data Classes          """
# 1. **Transportation and Logistics** Manufacturing Supply Transportation Chain
transportation_classes = {
    "Factory": ["Electronics Plant", "Textile Mill", "Auto Factory", "Chemical Plant", "Food Processing Plant"],
    "Assembly": ["Component Assembly", "Quality Check", "Packaging Line", "Labeling", "Test Facility"],
    "Warehouse": ["Regional Warehouse A", "Central Warehouse", "Cold Storage", "Spare Parts Depot", "Bulk Storage"],
    "Distribution Center": ["North Hub", "South Hub", "East Hub", "West Hub", "Express Sorting Center"],
    "Retail": ["City Supermarket", "Neighborhood Store", "Online Fulfillment", "Outlet Mall", "Convenience Store"]
}
# 2. **Tourism and Hospitality** Global Tourism Value Chain
tourism_classes = {
    "Source Regions": ["USA", "China", "Germany", "UK", "France", "Australia", "Japan"],
    "Tourism Cities": ["Paris", "Bangkok", "New York", "Tokyo", "Rome", "Barcelona", "Istanbul"],
    "Attractions": ["Museum", "Theme Park", "Historical Site", "Beach", "Shopping District", "Cultural Festival", "Hiking Trail"],
    "Hotels": ["Luxury Hotel", "Boutique Hotel", "Budget Hotel", "Resort", "Hostel", "Business Hotel"],
    "Spending Types": ["Accommodation", "Dining", "Transportation", "Shopping", "Entertainment", "Tours"]
}
# 3. **Business and Finance** Investment Decision-Making Routine
business_classes = {
    "Investment Types": ["Stocks", "Bonds", "Real Estate", "Commodities", "Cryptocurrency"],
    "Market Segments": ["Retail Investors", "Institutional Investors", "Hedge Funds", "Pension Funds"],
    "Investment Strategies": ["Growth Investing", "Value Investing", "Day Trading", "Index Fund Investing"],
    "Risk Levels": ["Low Risk", "Moderate Risk", "High Risk"],
    "Returns": ["Negative Return", "Low Return", "Moderate Return", "High Return"]
}
# 4. **Real Estate and Housing Market** Real Estate Transaction Ecosystem
real_estate_classes = {
    "Property Types": ["Single Family Home", "Condo", "Townhouse", "Multi-family Unit", "Commercial Property"],
    "Location Types": ["Urban", "Suburban", "Rural"],
    "Buyer Types": ["First-time Buyer", "Investor", "Retiree", "Family"],
    "Financing Options": ["Mortgage", "Cash Purchase", "Lease-to-Own"],
    "Market Conditions": ["Buyer's Market", "Seller's Market", "Neutral Market"]
}
# 5. **Healthcare and Health** Healthcare Service Ecosystem
healthcare_classes = {
    "Healthcare Providers": ["Hospitals", "Clinics", "Pharmacies", "Nursing Homes", "Rehabilitation Centers"],
    "Medical Specialties": ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology", "Dermatology"],
    "Treatment Types": ["Surgery", "Medication", "Therapy", "Preventive Care", "Emergency Care"],
    "Patient Demographics": ["Children", "Adults", "Seniors", "Pregnant Women", "Chronic Patients"],
    "Outcomes": ["Full Recovery", "Improved Condition", "Stable Condition", "Worsened Condition", "Deceased"]
}
# 6. **Retail and E-commerce** Omni-channel Retail Purchase Ecosystem
retail_classes = {
    "Product Categories": ["Electronics", "Clothing", "Home Appliances", "Books", "Groceries", "Beauty Products", "Toys"],
    "Customer Segments": ["Teens", "Young Adults", "Adults", "Seniors", "Parents", "Professionals", "Students"],
    "Channels": ["Online Store", "Mobile App", "Physical Store", "Third-Party Marketplace"],
    "Marketing Campaigns": ["Holiday Promo", "Flash Sale", "Email Newsletter", "Social Media Ad", "Referral Program", "Influencer Campaign"],
    "Regions": ["North America", "Europe", "Asia", "South America", "Middle East", "Africa", "Oceania"]
}
# 7. **Human Resources and Employee Management** Employee Management System
employee_classes = {
    "Recruitment Sources": ["Job Boards", "Referrals", "Social Media", "Recruitment Agencies", "University Career Fairs"],
    "Interview Stages": ["Phone Screen", "Technical Interview", "HR Interview", "Final Interview"],
    "Onboarding Steps": ["Document Verification", "Orientation", "Training Program", "Mentorship Assignment"],
    "Employee Types": ["Full-time", "Part-time", "Contractor", "Intern"],
    "Performance Ratings": ["Excellent", "Good", "Average", "Below Average"]
}
# 8. **Sports and Entertainment** Sports Event Framework
sports_entertainment_classes = {
    "Sports Types": ["Football", "Basketball", "Tennis", "Cricket", "Baseball", "Hockey", "Golf"],
    "Event Types": ["League Match", "Friendly Match", "Tournament", "Exhibition", "Playoffs", "Finals"],
    "Audience Segments": ["Local Fans", "International Fans", "VIP Guests", "Families", "Corporate Groups"],
    "Revenue Streams": ["Ticket Sales", "Merchandise", "Broadcast Rights", "Sponsorships", "Concessions"],
    "Engagement Channels": ["Social Media", "Live Streaming", "Fan Meetups", "Mobile Apps", "Fantasy Leagues"]
}
# 9. **Education and Academics** Education System Framework
education_academics_classes = {
    "Student Backgrounds": ["Urban", "Rural", "International", "First-Generation", "Low-Income", "Transfer Students"],
    "Enrollment Channels": ["Entrance Exams", "Direct Admission", "Scholarships", "Exchange Programs", "Online Applications"],
    "Institution Types": ["Public School", "Private School", "Community College", "University", "Online Institution"],
    "Support Services": ["Academic Advising", "Counseling", "Career Services", "Tutoring Centers", "Financial Aid"],
    "Post-Education Paths": ["Employment", "Graduate Studies", "Gap Year", "Entrepreneurship", "Certification Programs"]
}
# 10. **Food and Beverage Industry** Food Production and Consumption Chain
food_beverage_classes = {
    "Food Sources": ["Farms", "Fisheries", "Ranches", "Orchards", "Greenhouses"],
    "Processing Facilities": ["Slaughterhouses", "Canneries", "Dairies", "Bakeries", "Beverage Plants"],
    "Distribution Channels": ["Wholesalers", "Retailers", "Online Platforms", "Farmers' Markets", "Exporters"],
    "Consumption Venues": ["Restaurants", "Cafes", "Homes", "Schools", "Workplaces"],
    "Waste Management": ["Composting", "Recycling", "Landfills", "Food Banks", "Animal Feed"]
}
# 11. **Science and Engineering** Research and Development Ecosystem
science_eng_classes = {
    "Research Fields": ["Artificial Intelligence", "Quantum Computing", "Biotechnology", "Nanotechnology", "Renewable Energy", "Space Exploration", "Robotics"],
    "Institution Types": ["Research Universities", "Corporate Labs", "Government Agencies", "Startups", "Non-profit Organizations"],
    "Funding Sources": ["Federal Grants", "Private Investment", "Corporate Sponsorship", "University Budget", "Crowdfunding"],
    "Research Outcomes": ["Patents", "Published Papers", "Prototypes", "Open-source Projects", "Commercial Products"],
    "Impact Areas": ["Healthcare", "Environment", "Defense", "Education", "Industrial Automation"]
}
# 12. **Agriculture and Food Production** Agricultural Value Chain
agriculture_classes = {
    "Farming": ["Soil Preparation", "Seeding", "Irrigation", "Fertilization", "Pest Control", "Harvesting"],
    "Processing": ["Sorting", "Cleaning", "Packaging", "Freezing", "Canning", "Grinding"],
    "Distribution": ["Storage", "Transportation", "Wholesale", "Retail Delivery"],
    "Import & Export": ["Import Inspection", "Customs Clearance", "Export Certification", "International Shipping", "Trade Agreements"],
    "Consumption": ["Retail Purchase", "Food Preparation", "Consumption", "Food Waste"]
}
# 13. **Energy and Utilities** Carbon Emission Flow
carbon_classes = {
    "Emission Sources": ["Coal Combustion", "Oil Refining", "Natural Gas", "Cement Production", "Deforestation", "Livestock"],
    "Sector Distribution": ["Power Generation", "Manufacturing", "Transportation", "Agriculture", "Residential", "Waste Management"],
    "Transportation Methods": ["Pipeline", "Maritime Shipping", "Rail Transport", "Road Freight", "Air Cargo"],
    "Mitigation Measures": ["Carbon Capture", "Renewable Energy", "Energy Efficiency", "Afforestation", "None"],
    "Final Disposition": ["Atmosphere", "Carbon Storage", "Carbon Trading", "Offset Projects"]
}
# 14. **Cultural Trends and Influences** Cultural Products Influence Flow
culture_classes = {
    "Cultural Origins": ["Asia", "Europe", "Africa", "Americas", "Oceania"],
    "Art Forms": ["Painting", "Sculpture", "Music", "Dance", "Literature", "Cinema"],
    "Cultural Mediums": ["Museums", "Festivals", "Theaters", "Digital Platforms", "Workshops"],
    "Audience Types": ["Local Communities", "Tourists", "Scholars", "Artists", "General Public"],
    "Impact Areas": ["Education", "Tourism", "Economy", "Social Integration", "Innovation"]
}

# 15. **Social Media and Digital Media and Streaming** Digital Media Consumption Ecosystem
social_media_streaming_classes = {
    "User Demographics": ["Teens", "Young Adults", "Adults", "Seniors", "Content Creators", "Influencers"],
    "Platform Types": ["Social Networks", "Video Sharing", "Music Streaming", "Live Streaming", "Podcast Platforms", "News Aggregators"],
    "Content Formats": ["Short Videos", "Livestreams", "Podcasts", "Reels & Stories", "Music Tracks", "Articles"],
    "Engagement Methods": ["Likes", "Shares", "Comments", "Subscriptions", "Live Chats", "Donations"],
    "Monetization Models": ["Ad Revenue", "Sponsorships", "Subscriptions", "Pay-Per-View", "Virtual Gifts", "Merch Sales"]
}

"""              End Sankey Data Classes          """

def generate_sankey_data(num_files, theme_name, data_unit, min_startVal, max_startVal, classes, file_prefix):
    for j in range(1, num_files + 1):
        # Random number of stages (2-5 for simpler flows)
        num_classes = random.randint(2, 5)
        selected_class_names = random.sample(list(classes.keys()), num_classes)

        # Build flow structure
        flow_structure = []
        for class_name in selected_class_names:
            num_substeps = random.randint(2, min(7, len(classes[class_name])))
            steps = random.sample(classes[class_name], num_substeps)
            flow_structure.append(steps)

        # Distribution patterns
        distribution_mode = random.choice(["random", "linear", "long_tail", "normal"])
        start_value = random.randint(min_startVal, max_startVal)  # Represents number of students

        # Initialize source nodes
        source_nodes = flow_structure[0]
        num_sources = len(source_nodes)

        if distribution_mode == "random":
            weights = np.random.rand(num_sources)
        elif distribution_mode == "linear":
            weights = np.linspace(1, 0.3, num_sources)
        elif distribution_mode == "long_tail":
            weights = 1 / np.linspace(1, num_sources, num_sources)
        elif distribution_mode == "normal":
            mean = num_sources / 2
            std_dev = num_sources / 4
            weights = np.exp(-0.5 * ((np.arange(num_sources) - mean) / std_dev) ** 2)

        weights /= weights.sum()
        source_values = [int(start_value * w) for w in weights]
        diff = start_value - sum(source_values)
        source_values[0] += diff
        node_values = {node: val for node, val in zip(source_nodes, source_values)}

        sankey_data = []

        # Layer transitions
        for i in range(len(flow_structure) - 1):
            current_layer = flow_structure[i]
            next_layer = flow_structure[i + 1]
            next_layer_values = {}

            num_targets = len(next_layer)
            if distribution_mode == "random":
                weights = np.random.rand(num_targets)
            elif distribution_mode == "linear":
                weights = np.linspace(1, 0.3, num_targets)
            elif distribution_mode == "long_tail":
                weights = 1 / np.linspace(1, num_targets, num_targets)
            elif distribution_mode == "normal":
                mean = num_targets / 2
                std_dev = num_targets / 4
                weights = np.exp(-0.5 * ((np.arange(num_targets) - mean) / std_dev) ** 2)

            weights /= weights.sum()
            total_input = sum(node_values[n] for n in current_layer)
            next_values = [int(total_input * w) for w in weights]
            diff = total_input - sum(next_values)
            next_values[0] += diff

            for tgt, val in zip(next_layer, next_values):
                next_layer_values[tgt] = val

            for tgt in next_layer:
                target_val = next_layer_values[tgt]
                avail_values = [node_values[src] for src in current_layer]
                total_avail = sum(avail_values)
                src_weights = [v / total_avail for v in avail_values]

                for src, w in zip(current_layer, src_weights):
                    flow = int(target_val * w)
                    sankey_data.append((src, tgt, max(flow, 1)))
                    node_values[src] -= flow
                    node_values[src] = max(node_values[src], 0)

            node_values = next_layer_values

        # Save CSV
        filename = f"{file_prefix}_{j}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{theme_name}, {data_unit}, {distribution_mode}\n")
            f.write("Source,Target,Value\n")
            for src, tgt, val in sankey_data:
                f.write(f"{src},{tgt},{val}\n")

        print(f"✅ Generated: {filepath} | Distribution: {distribution_mode}")

sankey_data_configs = {
    "transportation_supply_chain": {
        "number": 1,
        "theme_name": "Manufacturing Supply Transportation Chain",
        "data_unit": "Goods",
        "min": 1000,
        "max": 5000,
        "classes": transportation_classes
    },
    "tourism_value_chain": {
        "number": 1,
        "theme_name": "Global Tourism Value Chain",
        "data_unit": "Million Tourists",
        "min": 10000,
        "max": 100000,
        "classes": tourism_classes
    },
    "investment_decision_making": {
        "number": 1,
        "theme_name": "Investment Decision-Making Routine",
        "data_unit": "$",
        "min": 50000,
        "max": 100000,
        "classes": business_classes
    },
    "real_estate_transaction": {
        "number": 1,
        "theme_name": "Real Estate Transaction Ecosystem",
        "data_unit": "Properties",
        "min": 500,
        "max": 3000,
        "classes": real_estate_classes
    },
    "healthcare_service": {
        "number": 1,
        "theme_name": "Healthcare Service Ecosystem",
        "data_unit": "Patients",
        "min": 1000,
        "max": 50000,
        "classes": healthcare_classes
    },
    "retail_purchase": {
        "number": 1,
        "theme_name": "Omni-channel Retail Purchase Ecosystem",
        "data_unit": "Purchases",
        "min": 5000,
        "max": 200000,
        "classes": retail_classes
    },
    "employee_management": {
        "number": 1,
        "theme_name": "Employee Management System",
        "data_unit": "Employees",
        "min": 500,
        "max": 10000,
        "classes": employee_classes
    },
    "sports_event": {
        "number": 1,
        "theme_name": "Sports Event Framework",
        "data_unit": "Events",
        "min": 1000,
        "max": 200000,
        "classes": sports_entertainment_classes
    },
    "education_system": {
        "number": 1,
        "theme_name": "Education System Framework",
        "data_unit": "Students",
        "min": 1000,
        "max": 50000,
        "classes": education_academics_classes
    },
    "food_consumption": {
        "number": 1,
        "theme_name": "Food Production and Consumption Chain",
        "data_unit": "Products",
        "min": 10000,
        "max": 200000,
        "classes": food_beverage_classes
    },
    "research_development": {
        "number": 1,
        "theme_name": "Research and Development Ecosystem",
        "data_unit": "Projects",
        "min": 50,
        "max": 2000,
        "classes": science_eng_classes
    },
    "agriculture_value_chain": {
        "number": 1,
        "theme_name": "Agricultural Value Chain",
        "data_unit": "Crops",
        "min": 1000,
        "max": 100000,
        "classes": agriculture_classes
    },
    "carbon_emission": {
        "number": 1,
        "theme_name": "Carbon Emission Flow",
        "data_unit": "Tons",
        "min": 1000,
        "max": 500000,
        "classes": carbon_classes
    },
    "cultural_influence": {
        "number": 1,
        "theme_name": "Cultural Influence Flow",
        "data_unit": "Cultural Products",
        "min": 100,
        "max": 5000,
        "classes": culture_classes
    },
    "social_media_streaming": {
        "number": 1,
        "theme_name": "Digital Media Consumption Ecosystem",
        "data_unit": "Views",
        "min": 1000,
        "max": 5000000,
        "classes": social_media_streaming_classes
    }
}

if __name__ == '__main__':
    for prefix, config in sankey_data_configs.items():
        generate_sankey_data(
            config["number"],
            config["theme_name"],
            config["data_unit"],
            config["min"],
            config["max"],
            config["classes"],
            prefix
        )

