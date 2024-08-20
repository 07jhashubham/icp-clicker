import React from "react";
import { Box, Grid, Typography, IconButton } from "@mui/material";
import TwitterIcon from "@mui/icons-material/Twitter";
import InstagramIcon from "@mui/icons-material/Instagram";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import TelegramIcon from "@mui/icons-material/Telegram";
import WallpaperIcon from "@mui/icons-material/Wallpaper";
import ImageLabeler from "./ImageLabeler";
import { Link } from "react-router-dom"; // Import Link from react-router-dom
export default function Task() {
  const socialMedia = [
    {
      name: "Twitter",
      icon: <TwitterIcon />,
      link: "https://twitter.com",
      points: "+200",
    },
    {
      name: "Instagram",
      icon: <InstagramIcon />,
      link: "https://instagram.com",
      points: "+100",
    },
    {
      name: "LinkedIn",
      icon: <LinkedInIcon />,
      link: "https://linkedin.com",
      points: "+50",
    },
    {
      name: "Telegram",
      icon: <TelegramIcon />,
      link: "https://telegram.org",
      points: "+500",
    },
  ];

  return (
    <Box
      sx={{
        color: "#fff",
        minHeight: "100vh",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Arial, sans-serif",
      }}
    >
      {/* Single Box Above the Title */}
      <Box
        sx={{
          backgroundColor: "#282235",
          borderRadius: "12px",
          padding: "30px 30px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          boxShadow: "0 8px 16px rgba(0, 0, 0, 0.4)",
          transition: "transform 0.3s, background-color 0.3s",
          width: "500px",
          marginBottom: "40px",
          "&:hover": {
            transform: "scale(1.05)",
            backgroundColor: "#34304a",
          },
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <IconButton
            href="https://twitter.com"
            target="_blank"
            sx={{
              backgroundColor: "#00e676",
              color: "#fff",
              borderRadius: "50%",
              padding: "16px",
              marginRight: "20px",
              "&:hover": {
                backgroundColor: "#00c853",
              },
            }}
          >
            <WallpaperIcon />
          </IconButton>
          <Typography
            variant="h6"
            sx={{ color: "#ffffff", fontWeight: "500" }}
            component={Link} // Use Link component
            to="/image-label" // Replace with your actual route
          >
            Label Data
          </Typography>
        </Box>
        <Typography variant="h5" sx={{ fontWeight: "bold", color: "#ffeb3b" }}>
          +2000
        </Typography>
      </Box>

      {/* Title */}
      <Typography
        variant="h3"
        sx={{
          marginBottom: "40px",
          textAlign: "center",
          color: "#00e676",
          fontWeight: "bold",
          textTransform: "uppercase",
          letterSpacing: "2px",
        }}
      >
        Connect With Us
      </Typography>

      {/* Social Media Boxes */}
      <Grid container direction="column" spacing={3}>
        {socialMedia.map((media, index) => (
          <Grid item key={index}>
            <Box
              sx={{
                backgroundColor: "#282235",
                borderRadius: "12px",
                padding: "20px 30px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.4)",
                transition: "transform 0.3s, background-color 0.3s",
                "&:hover": {
                  transform: "scale(1.05)",
                  backgroundColor: "#34304a",
                },
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center" }}>
                <IconButton
                  href={media.link}
                  target="_blank"
                  sx={{
                    backgroundColor: "#00e676",
                    color: "#fff",
                    borderRadius: "50%",
                    padding: "16px",
                    marginRight: "20px",
                    "&:hover": {
                      backgroundColor: "#00c853",
                    },
                  }}
                >
                  {media.icon}
                </IconButton>
                <Typography
                  variant="h6"
                  sx={{ color: "#ffffff", fontWeight: "500" }}
                >
                  {media.name}
                </Typography>
              </Box>
              <Typography
                variant="h5"
                sx={{ fontWeight: "bold", color: "#ffeb3b" }}
              >
                {media.points}
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
