using AutoMapper;
using ML.API.Models.Housing;

namespace ML.API.Models.Mapping
{
    public class HousingProfile : Profile
    {
        public HousingProfile()
        {
            CreateMap<HousingRequest, HousingInput>()
                .ForMember(dest => dest.SquareFootage, opt => opt.MapFrom(src => src.SquareFootage))
                .ForMember(dest => dest.NumBedrooms, opt => opt.MapFrom(src => src.NumBedrooms))
                .ForMember(dest => dest.NumBathrooms, opt => opt.MapFrom(src => src.NumBathrooms))
                .ForMember(dest => dest.YearBuilt, opt => opt.MapFrom(src => src.YearBuilt))
                .ForMember(dest => dest.LotSize, opt => opt.MapFrom(src => src.LotSize))
                .ForMember(dest => dest.GarageSize, opt => opt.MapFrom(src => src.GarageSize))
                .ForMember(dest => dest.NeighbourhoodQuality, opt => opt.MapFrom(src => src.NeighbourhoodQuality));
            CreateMap<HousingOutput, HousingResponse>()
                .ForMember(x => x.Price, x => x.MapFrom(x => x.HousePrice));
        }
    }
}
